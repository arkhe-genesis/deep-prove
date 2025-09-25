//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use crate::{
    IO, Proof, Prover, ProverContext,
    iop::context::VerifierContext,
    padding::PaddingMode,
    quantization::{InferenceObserver, IntoElement, ScalingStrategy},
    verify,
};
use anyhow::{Context as CC, ensure};
use ark_std::rand::Rng;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;
use tenstore::GenStore;
use tracing::{debug, info};
use transcript::BasicTranscript;

use crate::{
    Element, Shape, Tensor,
    layers::{Layer, provable::Evaluate},
    model::{InferenceTrace, Model},
    number::Number,
    padding::pad_model,
    parser::{
        gguf, json,
        llm::{LLMConfig, LLMTokenizer, Token},
    },
};

pub trait Observer<N: Number> {
    fn observe<E: ExtensionField>(&self, step: usize, trace: &InferenceTrace<'_, E, N>);
}

/// The main struct responsible for generating the trace and the proof related
/// to LLM proving. This requires a wrapper on top of the model to drive the
/// auto regressive loop correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Driver<N: Number> {
    pub(crate) model: Model<N>,
    config: LLMConfig,
    max_context: Option<usize>,
    padding_mode: PaddingMode,
}

/// The main struct responsible for verifying the proof related to the LLM proving.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub prover_ctx: ProverContext<E, PCS>,
    pub verifier_ctx: VerifierContext<E, PCS>,
    pub config: LLMConfig,
    pub max_context: Option<usize>,
}

impl<E, PCS> LLMContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMProof<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub proof: Proof<E, PCS>,
    /// Note the IO contains the _full_ input, e.g. the user input + the generated tokens
    pub io: IO<E>,
}

impl Driver<f32> {
    /// Loads a model from a gguf or json external file. It returns the raw model in float precision.
    pub fn load_external_model<S: AsRef<Path>>(path: S) -> anyhow::Result<Self> {
        // detect the type of the model info, either json or gguf depending on the file extension
        let (config, llm_model) = match path
            .as_ref()
            .extension()
            .unwrap_or_default()
            .to_str()
            .unwrap()
        {
            "json" => {
                let loader = json::FileTensorLoader::new_from_path(path)?;
                let config = LLMConfig::from_json(&loader)?;
                let llm_model = config.model_json(&loader)?;
                (config, llm_model)
            }
            "gguf" => {
                let loader = gguf::FileTensorLoader::from_path(path)?;
                let config = LLMConfig::from_content(&loader)?;
                let llm_model = config.model(&loader)?;
                (config, llm_model)
            }
            _ => anyhow::bail!(
                "Unsupported model file extension: {}",
                path.as_ref()
                    .extension()
                    .unwrap_or_default()
                    .to_str()
                    .unwrap()
            ),
        };

        // even though the llm runtime doesn't care about the model input shape, which is designed for "static" input shapes, we still
        // need to provide one.
        let init_user_shape = Shape::from(vec![1]);
        let model = llm_model.into_provable_model(&config, init_user_shape)?;
        Ok(Self {
            model,
            config,
            max_context: None,
            padding_mode: PaddingMode::NoPadding,
        })
    }

    pub fn into_runnable_llm(mut self) -> anyhow::Result<Driver<Element>> {
        let numel = self.max_context.unwrap_or(self.config.context_length);
        let n_inputs = 1;
        let representative_inputs = (0..n_inputs)
            .map(|_| {
                self.random_sequence(numel)
                    .into_iter()
                    .map(|t| t.as_number())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.model.unpadded_input_shapes = vec![Shape::from(vec![numel])];
        self.model.input_shapes = vec![Shape::from(vec![numel, 1])];
        let (quantized_model, _md) =
            InferenceObserver::new_with_representative_input(vec![representative_inputs])
                .quantize(self.model, &mut GenStore::default())?;
        Ok(Driver {
            model: quantized_model,
            config: self.config,
            max_context: self.max_context,
            padding_mode: PaddingMode::NoPadding,
        })
    }

    /// Transform the model into a provable llm model with quantization and padding done.
    /// The result can be serialized and deserialized at will to serve inference+proving for this model.
    pub fn into_provable_llm(self) -> anyhow::Result<Driver<Element>> {
        let quantized_llm = self.into_runnable_llm()?;
        quantized_llm.pad_model()
    }

    pub fn run<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<f32>>,
    ) -> anyhow::Result<InferenceTrace<'_, E, f32>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let user_len = input.len();

        ensure!(
            user_len < self.config.context_length - 1,
            "Input sequence length must be less than the context length"
        );
        let input_tokens = input
            .into_iter()
            .map(|t| t.as_number::<f32>())
            .collect::<Vec<_>>();

        let tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone());
        let shape = tensor.shape().clone();
        let mut store = GenStore::default();

        let trace = self
            .model
            .run::<E>(&[tensor], Some(vec![shape]), &mut store)?;

        if let Some(ref obs) = observer {
            obs.observe(0, &trace);
        }

        Ok(trace)
    }
}

impl<N: Number + Serialize + for<'a> Deserialize<'a>> Driver<N>
where
    Layer<N>: Evaluate<N>,
{
    pub fn new(
        model: Model<N>,
        config: LLMConfig,
        max_context: Option<usize>,
        padding_mode: PaddingMode,
    ) -> Self {
        Self {
            model,
            config,
            max_context,
            padding_mode,
        }
    }
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }

    pub fn random_sequence(&self, len: usize) -> Vec<Token> {
        let mut rng = crate::rng_from_env_or_random();
        (0..len)
            .map(|_| Token::from(rng.gen_range(0..self.config.vocab_size)))
            .collect()
    }

    /// Runs take the _already_ tokenized input and run the model until the maximum sequence length is reached OR until a eos token is generated.
    /// The returned trace contains the _whole_ sequence.
    fn run_internal<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<N>>,
    ) -> anyhow::Result<InferenceTrace<'_, E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let eos_token: N = self.config.eos_token.as_number();
        let user_len = input.len();
        // -1 because we at least want to generate ONE token
        ensure!(
            user_len < self.config.context_length - 1,
            "Input sequence length must be less than the context length"
        );
        let input_tokens = input
            .into_iter()
            .map(|t| t.as_number::<N>())
            .collect::<Vec<_>>();

        let mut tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone());
        let max_window = self.max_context.unwrap_or(self.config.context_length);

        ensure!(
            tensor
                .get_data()
                .iter()
                .all(|t| t.to_usize() < self.config.vocab_size),
            "Input tokens must be less than the vocabulary size"
        );
        let mut full_tokens = tensor.get_data().to_vec();
        let mut unpadded_seq_len = user_len;
        // convert the input to the correct number type and add a dimension to make it 2d, because the embeddings layer expects a 2d tensor
        // This means we're padding the input to the right size (e.g. next power of two)
        while unpadded_seq_len <= max_window {
            info!(
                "Running iteration {} with input tensor {:?}",
                unpadded_seq_len,
                tensor.shape()
            );
            let mut store = GenStore::default();
            let trace = if let PaddingMode::NoPadding = self.padding_mode {
                self.model
                    // TODO: make it re-usable at least for the static weights
                    .run::<E>(
                        &[tensor.clone()],
                        Some(vec![tensor.shape().clone()]),
                        &mut store,
                    )
            } else {
                let unpadded_shape = tensor.shape().clone();
                let padded = tensor.pad_next_power_of_two();
                info!("LLM: running model with unpadded shape: {unpadded_shape:?}");
                self.model
                    .run::<E>(&[padded], Some(vec![unpadded_shape]), &mut store)
            }
            .context(format!(
                "runng the {} iteration loop",
                unpadded_seq_len - user_len
            ))?;
            ensure!(
                trace.output.len() == 1,
                "expected 1 output, got {}",
                trace.output.len()
            );
            let output = trace.outputs().unwrap().into_iter().last().unwrap();
            // We take the last token before the padding
            let output_tokens_len = output.get_data().len();
            let last_token = if output_tokens_len == 1 {
                *output.get_data().last().expect("last token must exist")
            } else {
                output.get_data()[unpadded_seq_len - 1]
            };
            tensor = Tensor::new(Shape::new(vec![1]), vec![last_token]);
            full_tokens.push(last_token);
            if last_token == eos_token {
                break;
            }
            if let Some(ref obs) = observer {
                obs.observe(unpadded_seq_len - user_len, &trace);
            }
            unpadded_seq_len += 1;
        }
        // 1. input construction: we remove the last token since it's either the eos token or the max token
        // we need to regenerate a full trace with the correct padding
        full_tokens.pop();
        // and we take only the part that corresponds to the _generated_ tokens
        full_tokens.splice(..user_len, input_tokens);
        let input_len = full_tokens.len();
        tensor = Tensor::new(Shape::new(vec![input_len]), full_tokens.clone());
        // 2. padding: we pad the input to the expected shape of the model
        let target_padded_shape = vec![max_window.next_power_of_two()].into();
        if let PaddingMode::Padding = self.padding_mode {
            tensor.pad_to_shape(target_padded_shape)
        };
        // 3. model resetting: we need to _reset_ the cache of every QKV layer in the model - that's because we only
        // expect 1 token to be generated at a time after the first inference.
        self.model.reset();
        // 4. rerun to have a "clean" trace
        info!("Running last iteration (heavy) with {input_len} tokens");

        let mut store = GenStore::default();
        let trace = self.model.run::<E>(
            &[tensor],
            Some(vec![Shape::new(vec![input_len])]),
            &mut store,
        )?;
        for i in user_len..input_len {
            assert_eq!(
                trace.outputs().unwrap()[0].get_data()[i - 1],
                trace.inputs().unwrap()[0].get_data()[i],
                "Failed for {i}, input: {:?}, output: {:?}",
                trace.inputs().unwrap()[0],
                trace.outputs().unwrap()[0]
            );
        }
        Ok(trace)
    }
}

impl Driver<Element> {
    pub fn run<E>(
        &self,
        input: Vec<Token>,
        observer: Option<impl Observer<Element>>,
    ) -> anyhow::Result<InferenceTrace<'_, E, Element>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_internal::<E>(input, observer)
    }

    pub fn pad_model(mut self) -> anyhow::Result<Self> {
        let numel = self.max_context.unwrap_or(self.config.context_length);
        self.model.input_shapes = vec![Shape::from(vec![numel, 1]).next_power_of_two()];
        let model = pad_model(self.model)?;
        Ok(Self {
            model,
            config: self.config,
            max_context: self.max_context,
            padding_mode: PaddingMode::Padding,
        })
    }

    /// Get the size of the maximum polynomial employed when proving the LLM inference for the given `input_len`.
    /// This size determines which context to be used to run the proving among all the possible proving contexts
    /// for the given model
    pub fn get_max_poly_size_for_input<E: ExtensionField>(
        &self,
        input_len: usize,
    ) -> anyhow::Result<usize> {
        self.model
            .compute_max_poly_size::<E>(&[vec![input_len.next_power_of_two()].into()])
    }

    /// Compute the set of contexts necessary for all the possible input shapes of the LLM.
    /// It returns a `HashMap` which associates a given context to the maximum polynomial size supported
    /// by that context. The proper context to be used for a given input size `input_len` is found in the
    /// entry `HashMap` returned by this method identified by the key `self.get_max_poly_size_for_input(input_len)`
    pub fn compute_all_contexts<E, PCS>(&self) -> anyhow::Result<LLMContext<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        // compute shapes for all possible input sequence lengths
        let max_input_shapes = vec![Shape::new(vec![
            self.config.context_length.next_power_of_two(),
        ])];
        ensure!(
            max_input_shapes.len() == 1,
            "Expected 1 input shape in LLM model, found {}",
            max_input_shapes.len()
        );
        let max_input_length = max_input_shapes[0].numel();
        ensure!(max_input_length.is_power_of_two());

        let (prover_ctx, verifier_ctx) = self
            .model
            .generate_contexts_for_input_shapes(max_input_shapes)?;
        Ok(LLMContext {
            prover_ctx,
            verifier_ctx,
            config: self.config.clone(),
            max_context: self.max_context,
        })
    }

    /// Create the prover & verifier for a given model
    pub fn context<E, PCS>(&self) -> anyhow::Result<LLMContext<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        debug!(
            "Generating context for model with {} layers",
            self.model.nodes.len()
        );
        let (prover_ctx, verifier_ctx) = self.model.generate_contexts()?;
        Ok(LLMContext {
            prover_ctx,
            verifier_ctx,
            config: self.config.clone(),
            // The verifier should put itself the max context here
            max_context: None,
        })
    }
    pub fn prove<E, PCS>(
        &self,
        ctx: &LLMContext<E, PCS>,
        trace: InferenceTrace<'_, E, Element>,
    ) -> anyhow::Result<LLMProof<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let mut tr: BasicTranscript<E> = BasicTranscript::new(b"model");
        let prover: Prover<'_, '_, E, _, _> = Prover::new(&ctx.prover_ctx, &mut tr);
        let io = trace.to_verifier_io()?;
        info!("Proving the trace");
        let proof = prover.prove(&trace).expect("unable to generate proof");
        info!("Proof generated");
        Ok(LLMProof { proof, io })
    }
}

impl<E, PCS> LLMContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn verify(&self, proof: LLMProof<E, PCS>, user_input: Vec<Token>) -> anyhow::Result<()>
    where
        E: ExtensionField + Serialize + DeserializeOwned + Number,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // 0. check the size of the output
        let output = proof.io.output[0].clone();
        let padded_max_len = output.shape().numel();
        let max_window = self.max_context.unwrap_or(self.config.context_length);

        // in any case, the output needs to be less than the max context length
        ensure!(
            padded_max_len <= max_window.next_power_of_two(),
            "output length is greater than the padded maximum context length"
        );
        // get the actual output length: could be either `max_window`, or when an eos token is found
        let eos_token = self.config.eos_token;
        let eos_token_found = output
            .get_data()
            .iter()
            .take(max_window) // consider only the first max_window tokens
            .skip(user_input.len() - 1) // the first user_input.len() - 1 are not meaningful
            .find_position(|token| token.to_element() as usize == usize::from(eos_token));
        let unpadded_output_len = if let Some(pos) = &eos_token_found {
            pos.0
        } else {
            max_window
        };

        // 1. verify the proof it self
        let mut tr: BasicTranscript<E> = BasicTranscript::new(b"model");
        let prover_input = proof.io.input[0].clone();
        let prover_output = proof.io.output[0].clone();
        verify::<_, _, _>(&self.verifier_ctx, proof.proof, proof.io, &mut tr)?;
        // 2. verify the sequentiality of the output: from the first newly generated token to the last
        // but without including the padding.
        // output is [seq_len] where []
        let seq_len = user_input.len();
        ensure!(
            prover_input.get_data()[..seq_len]
                .iter()
                .zip(user_input[..seq_len].iter())
                .all(|(a, b)| a.to_element().to_usize() == b.0),
            "user input not the same"
        );

        #[allow(clippy::needless_range_loop)]
        for i in seq_len - 1..unpadded_output_len - 1 {
            // we check the next input token is the one generated by this "row" of the input
            ensure!(
                prover_input.get_data()[i + 1] == prover_output.get_data()[i],
                "next input token is not the one generated by this row pos i {}: {:?} != {:?}",
                i,
                prover_input.get_data(),
                prover_output.get_data()
            );
        }
        Ok(())
    }
}

pub struct LLMTokenizerObserver<'a, T: LLMTokenizer> {
    pub input: String,
    pub tokenizer: &'a T,
}

impl<'a, N, T: LLMTokenizer> Observer<N> for LLMTokenizerObserver<'a, T>
where
    N: Number + Serialize + for<'b> Deserialize<'b>,
{
    fn observe<E: ExtensionField>(&self, step: usize, trace: &InferenceTrace<'_, E, N>) {
        let tensor = trace
            .output
            .last()
            .unwrap()
            .hydrate(trace.store.clone())
            .expect("hydration failed");
        let output_tokens_len = tensor.get_data().len();
        let input_tokens = self.tokenizer.tokenize(&self.input);
        let new_token = if output_tokens_len == 1 {
            *tensor.get_data().last().expect("last token must exist")
        } else {
            tensor.get_data()[input_tokens.len() + step - 1]
        };

        // let new_token = tensor.get_data().last().unwrap();
        let new_token = Token::from(new_token.to_usize());
        let new_text = self.tokenizer.detokenize(&[new_token]);
        debug!(
            "seq_len {}: new token: {:?}\n\t-{}", //\n\t-{:?}",
            step,
            &new_token,
            (self.input.clone() + &new_text).trim(),
            // tensor.get_data()
        );
    }
}

#[cfg(test)]
mod test {
    use crate::{
        init_test_logging,
        parser::{
            file_cache,
            gguf::tests::{GEMMA3_Q8, GPT2_Q8_0},
            llm::{HFTokenizer, Token},
        },
        testing::Pcs,
    };

    use super::*;
    use ff_ext::GoldilocksExt2;

    #[test]
    fn test_llm_driver_prove() -> anyhow::Result<()> {
        const MAX_CONTEXT: usize = 10;
        init_test_logging("debug");

        // Load the model file
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        // let model_path = "assets/scripts/llms/toy_gpt2.gguf";
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GPT2_Q8_0}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        let (driver, ctx): (_, LLMContext<GoldilocksExt2, Pcs<GoldilocksExt2>>) =
            file_cache::deserialize_or_create_with(&cache_filename, || {
                let driver = Driver::load_external_model(&model_path)?
                    .with_max_context(MAX_CONTEXT)
                    .into_provable_llm()?;

                let ctx = driver
                    .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
                    .with_max_context(MAX_CONTEXT);

                Ok((driver, ctx))
            })?;

        // Generate the trace
        let sentence = "The sky is";
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;

        // Prove the trace
        let proof = driver.prove(&ctx, trace)?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        ctx.verify(proof, user_tokens)?;
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference() -> anyhow::Result<()> {
        init_test_logging("debug");
        const PRUNED_GPT2: &str = "gpt2.Q2_K.gguf";
        let model_path = file_cache::from_cache(PRUNED_GPT2)?;
        // let model_path = "assets/scripts/llms/toy_gpt2.gguf";
        let driver = Driver::load_external_model(&model_path)?
            .with_max_context(6)
            .into_runnable_llm()?;
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {user_tokens:?}");
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let _output = trace
            .outputs()
            .unwrap()
            .last()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        // let output = detokenize(&tokenizer, &output);
        // println!("{}", output);
        Ok(())
    }

    #[test]
    fn test_llm_gemma3() -> anyhow::Result<()> {
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let driver = Driver::load_external_model(&model_path)?.with_max_context(6);

        let driver = driver.into_runnable_llm()?;
        println!("LLM DRIVER: config: {:?}", driver.config);

        let sentence = "The sky is";
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let output = trace
            .outputs()
            .unwrap()
            .last()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output);
        println!("detokenized output: {output}");
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_gemma3() -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 8;
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GEMMA3_Q8}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        let (mut driver, ctx): (_, LLMContext<GoldilocksExt2, Pcs<GoldilocksExt2>>) =
            file_cache::deserialize_or_create_with(&cache_filename, || {
                let driver = Driver::load_external_model(&model_path)?
                    .with_max_context(MAX_CONTEXT)
                    .into_provable_llm()?;

                let ctx = driver
                    .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
                    .with_max_context(MAX_CONTEXT);

                Ok((driver, ctx))
            })?;

        println!("LLM DRIVER: config: {:?}", driver.config);
        // Generate the trace
        let sentence = "The sky is";
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);

        driver = driver.pad_model()?;

        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;

        // Prove the trace
        let proof = driver.prove(&ctx, trace)?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        ctx.verify(proof, user_tokens)?;
        Ok(())
    }
}
