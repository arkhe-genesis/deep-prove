//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use crate::{
    IO, Proof, Prover, ProverContext,
    graph::executor::{Executor, SequentialExecutor},
    iop::{
        chunking::{ChunkingStrategy, DefaultChunkingStrategy},
        context::VerifierContext,
        prover_graph::ProverGraphNode,
    },
    padding::PaddingMode,
    parser::{
        PipelineConfig, default_pipeline_config,
        llm::{LLMConfig, LLMTokenizer, Token, models::LLMModelLoader},
        to_quantized,
    },
    quantization::{InferenceObserver, InferenceTracker, IntoElement},
    tensor::TensorTypeParam,
    verify,
};
use anyhow::{Context as CC, ensure};
use ark_std::rand::Rng;
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use tracing::{debug, info};
use transcript::BasicTranscript;

use crate::{
    Element, Shape, Tensor,
    layers::{Layer, provable::Evaluate},
    model::{Model, Trace},
    number::Number,
};

pub trait Observer<N: TensorTypeParam> {
    fn observe<E: ExtensionField>(&self, step: usize, trace: &Trace<E, N>);
}

/// The main struct responsible for LLM proving.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMProverContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub prover_ctx: ProverContext<E, PCS>,
    pub config: LLMConfig,
    pub max_context: Option<usize>,
}

pub trait WithMaxContext {
    fn with_max_context(self, max_context: usize) -> Self;
}

impl<E, PCS> WithMaxContext for LLMProverContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

/// The main struct responsible to verify an LLM proof.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LLMVerifierContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub verifier_ctx: VerifierContext<E, PCS>,
    pub config: LLMConfig,
    pub max_context: Option<usize>,
}

impl<E, PCS> WithMaxContext for LLMVerifierContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

impl<E, PCS> WithMaxContext for (LLMProverContext<E, PCS>, LLMVerifierContext<E, PCS>)
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    fn with_max_context(self, max_context: usize) -> Self {
        (
            self.0.with_max_context(max_context),
            self.1.with_max_context(max_context),
        )
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

/// The main struct responsible for generating the trace and the proof related
/// to LLM proving. This requires a wrapper on top of the model to drive the
/// auto regressive loop correctly.
#[derive(Debug, Serialize, Deserialize)]
pub struct Driver<N> {
    pub model: Model<N>,
    config: LLMConfig,
    max_context: Option<usize>,
    padding_mode: PaddingMode,
}
impl Driver<f32> {
    /// Loads a model from a gguf, safetensors, or json external file. It returns the raw model in float precision.
    /// NOTE: the max_context is only there to hack around the creation of Rope to avoid loading the full matrix if we don't need it. That should
    /// be removed if when we remove this hack.
    pub fn load_from_model<DataFormat, M: LLMModelLoader<DataFormat>>(
        mut model_type: M,
        data_format: &DataFormat,
        max_context: Option<usize>,
    ) -> anyhow::Result<Self> {
        if let Some(max_context) = max_context {
            model_type = model_type.with_max_context_length(max_context);
        }
        let (model, config) = model_type.parse(data_format)?;
        Ok(Self {
            model,
            config,
            max_context,
            padding_mode: PaddingMode::NoPadding,
        })
    }

    /// Transform the model into a provable llm model with quantization and padding done.
    /// The result can be serialized and deserialized at will to serve inference+proving for this model.
    pub fn into_provable_llm<'a>(
        self,
        mut pipeline_config: Option<PipelineConfig<'a, InferenceObserver>>,
    ) -> anyhow::Result<Driver<Element>> {
        let numel = self.max_context.unwrap_or(self.config.context_length);
        let n_inputs = 1;
        let representative_inputs = (0..n_inputs)
            .map(|_| {
                self.random_sequence(numel)
                    .into_iter()
                    .map(|t| t.as_number::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantization_strategy =
            InferenceObserver::new_with_representative_input(vec![representative_inputs]);
        let mut conf = pipeline_config.take().unwrap_or_else(|| {
            // override the shapes to adhere to the expected input shape of the representative inputs
            default_pipeline_config()
        });

        if conf.input_shapes.is_none() {
            conf = conf.with_input_shapes(vec![Shape::from(vec![numel])]);
        }
        if conf.quant_strategy.is_none() {
            conf = conf.with_strategy(quantization_strategy);
        }
        let mut quantized_model = to_quantized(self.model, conf)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok(Driver {
            model: quantized_model,
            config: self.config,
            max_context: self.max_context,
            padding_mode: PaddingMode::Padding,
        })
    }

    pub fn run<E>(
        &self,
        input: &[Token],
        store: &mut GenStore,
        observer: Option<impl Observer<f32>>,
    ) -> anyhow::Result<Trace<E, f32>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_with_tracker(input, store, observer, None)
    }

    pub fn run_with_tracker<E>(
        &self,
        input: &[Token],
        store: &mut GenStore,
        observer: Option<impl Observer<f32>>,
        tracker: Option<&mut InferenceTracker>,
    ) -> anyhow::Result<Trace<E, f32>>
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
            .iter()
            .map(|t| t.as_number::<f32>())
            .collect::<Vec<_>>();

        let tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone())?;
        let trace = self
            .model
            .run_with_tracker::<E>(vec![tensor], tracker, store)?;

        if let Some(ref obs) = observer {
            obs.observe(0, &trace);
        }

        Ok(trace)
    }
}

impl<N: TensorTypeParam + Serialize + for<'a> Deserialize<'a>> Driver<N>
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

    /// Runs take the _already_ tokenized input and run the model until the
    /// maximum sequence length is reached OR until a eos token is generated.
    /// The returned trace contains the _whole_ sequence.
    fn run_internal<E>(
        &self,
        input: Vec<Token>,
        store: &mut GenStore,
        observer: Option<impl Observer<N>>,
        mut tracker: Option<&mut InferenceTracker>,
    ) -> anyhow::Result<Trace<E, N>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        let eos_token: N = self.config.eos_token.as_tensor_type_param();
        let user_len = input.len();
        // -1 because we at least want to generate ONE token
        ensure!(
            user_len < self.config.context_length - 1,
            "Input sequence length must be less than the context length"
        );
        let input_tokens = input
            .into_iter()
            .map(|t| t.as_tensor_type_param::<N>())
            .collect::<Vec<_>>();

        let mut tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone())?;
        let max_window = self.max_context.unwrap_or(self.config.context_length);
        ensure!(
            input_tokens.len() < max_window,
            "max window {} is smaller than prompt token count {}",
            max_window,
            input_tokens.len()
        );

        ensure!(
            tensor
                .get_data()
                .iter()
                .all(|t| Number::to_usize(t) < self.config.vocab_size),
            "Input tokens must be less than the vocabulary size"
        );
        let mut full_tokens = tensor.get_data().to_vec();
        let mut unpadded_seq_len = user_len;
        // convert the input to the correct number type and add a dimension to
        // make it 2d, because the embeddings layer expects a 2d tensor This
        // means we're padding the input to the right size (e.g. next power of
        // two)
        while unpadded_seq_len <= max_window {
            info!(
                "Running iteration {} with input tensor {:?}",
                unpadded_seq_len,
                tensor.shape()
            );
            let trace = if let PaddingMode::NoPadding = self.padding_mode {
                self.model
                    // TODO: make it re-usable at least for the static weights
                    .run_with_tracker::<E>(
                        vec![tensor.clone()],
                        // HACK: to not consume the option on repeated
                        if let Some(ref mut t) = tracker {
                            Some(*t)
                        } else {
                            None
                        },
                        store,
                    )
            } else {
                let unpadded_shape = tensor.shape().clone();
                let padded = tensor.pad_next_power_of_two();
                info!("LLM: running model with unpadded shape: {unpadded_shape:?}");
                self.model.run_with_tracker::<E>(
                    vec![padded],
                    // HACK: to not consume the option on repeated
                    if let Some(ref mut t) = tracker {
                        Some(*t)
                    } else {
                        None
                    },
                    store,
                )
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
            let output_handles = trace.outputs();
            let output_handle = output_handles.last().unwrap();
            let output = output_handle.tensor().unwrap();
            // We take the last token before the padding
            let output_tokens_len = output.get_data().len();
            let last_token = if output_tokens_len == 1 {
                *output.get_data().last().expect("last token must exist")
            } else {
                output.get_data()[unpadded_seq_len - 1]
            };
            tensor = Tensor::new(Shape::new(vec![1]), vec![last_token])?;
            full_tokens.push(last_token);
            if last_token == eos_token {
                break;
            }
            if let Some(ref obs) = observer {
                obs.observe(unpadded_seq_len - user_len, &trace);
            }
            unpadded_seq_len += 1;
        }
        // 1. input construction: we remove the last token since it's either the
        // eos token or the max token we need to regenerate a full trace with
        // the correct padding
        full_tokens.pop();
        // and we take only the part that corresponds to the _generated_ tokens
        full_tokens.splice(..user_len, input_tokens);
        let input_len = full_tokens.len();
        tensor = Tensor::new(Shape::new(vec![input_len]), full_tokens.clone())?;
        // 2. padding: we pad the input to the expected shape of the model
        let target_padded_shape = vec![max_window.next_power_of_two()].into();
        if let PaddingMode::Padding = self.padding_mode {
            tensor.pad_to_shape(target_padded_shape)?
        };
        // 3. model resetting: we need to _reset_ the cache of every QKV layer
        // in the model - that's because we only expect 1 token to be generated
        // at a time after the first inference.
        self.model.reset();
        // 4. rerun to have a "clean" trace
        info!("Running last iteration (heavy) with {input_len} tokens");

        let mut store = GenStore::default();
        let trace = self.model.run::<E>(vec![tensor], &mut store)?;
        for i in user_len..input_len {
            assert_eq!(
                trace.outputs()[0].tensor().unwrap().get_data()[i - 1],
                trace.inputs()[0].tensor().unwrap().get_data()[i],
                "Failed for {i}, input: {:?}, output: {:?}",
                trace.inputs()[0].tensor().unwrap(),
                trace.outputs()[0].tensor().unwrap(),
            );
        }
        Ok(trace)
    }
}

impl Driver<Element> {
    pub fn run<E>(
        &self,
        input: Vec<Token>,
        store: &mut GenStore,
        observer: Option<impl Observer<Element>>,
    ) -> anyhow::Result<Trace<E, Element>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_internal::<E>(input, store, observer, None)
    }

    pub fn run_with_tracker<E>(
        &self,
        input: Vec<Token>,
        store: &mut GenStore,
        observer: Option<impl Observer<Element>>,
        tracker: &mut InferenceTracker,
    ) -> anyhow::Result<Trace<E, Element>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
    {
        self.run_internal::<E>(input, store, observer, Some(tracker))
    }

    /// Compute the set of contexts necessary for all the possible input shapes of the LLM.
    /// It returns a `HashMap` which associates a given context to the maximum polynomial size supported
    /// by that context. The proper context to be used for a given input size `input_len` is found in the
    /// entry `HashMap` returned by this method identified by the key `self.get_max_poly_size_for_input(input_len)`
    pub fn compute_all_contexts<E, PCS>(
        &self,
    ) -> anyhow::Result<(LLMProverContext<E, PCS>, LLMVerifierContext<E, PCS>)>
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
        Ok((
            LLMProverContext {
                prover_ctx,
                config: self.config.clone(),
                max_context: self.max_context,
            },
            LLMVerifierContext {
                verifier_ctx,
                config: self.config.clone(),
                max_context: self.max_context,
            },
        ))
    }

    /// Create the prover & verifier for a given model
    pub fn context<E, PCS>(
        &self,
    ) -> anyhow::Result<(LLMProverContext<E, PCS>, LLMVerifierContext<E, PCS>)>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        debug!(
            "Generating context for model with {} layers",
            self.model.graph.node_count()
        );
        let (prover_ctx, verifier_ctx) = self.model.generate_contexts()?;
        Ok((
            LLMProverContext {
                prover_ctx,
                config: self.config.clone(),
                max_context: self.max_context,
            },
            LLMVerifierContext {
                verifier_ctx,
                config: self.config.clone(),
                // The verifier should put itself the max context here
                max_context: None,
            },
        ))
    }

    fn chunked_prove_local<'a, 'b, S, E, PCS, Ex>(
        &'a self,
        ctx: &'b LLMProverContext<E, PCS>,
        trace: Trace<E, Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
        executor_config: Ex::Config,
    ) -> anyhow::Result<LLMProof<E, PCS>>
    where
        S: ChunkingStrategy,
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
        Ex: Executor<ProverGraphNode<'b, 'a, E, BasicTranscript<E>, PCS>, usize>,
    {
        let io = trace.to_verifier_io()?;
        info!("Proving the trace");
        let proof = Prover::<E, BasicTranscript<E>, PCS>::chunked_prove_local::<_, Ex>(
            &ctx.prover_ctx,
            trace,
            num_chunks,
            chunking_strategy,
            &self.model,
            executor_config,
        )
        .expect("unable to generate proof");
        info!("Proof generated");
        Ok(LLMProof { proof, io })
    }

    /// Utility method to run `chunked_prove_local` with a predefined executor
    #[cfg(test)]
    pub(crate) fn chunked_prove_default_executor<S, E, PCS>(
        &self,
        ctx: &LLMProverContext<E, PCS>,
        trace: Trace<E, Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
    ) -> anyhow::Result<LLMProof<E, PCS>>
    where
        S: ChunkingStrategy,
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // NOTE: until https://github.com/Plonky3/Plonky3/pull/999 is fixed, we have
        // to use the sequential executor and not the threadpool executor.
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            num_chunks,
            chunking_strategy,
            (),
        )
    }

    pub fn prove<E, PCS>(
        &self,
        ctx: &LLMProverContext<E, PCS>,
        trace: Trace<E, Element>,
    ) -> anyhow::Result<LLMProof<E, PCS>>
    where
        E: ExtensionField + Serialize + DeserializeOwned,
        E::BaseField: Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'static,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            Some(1),
            DefaultChunkingStrategy::default(),
            (),
        )
    }
}

impl<E, PCS> LLMVerifierContext<E, PCS>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn verify(&self, proof: LLMProof<E, PCS>, user_input: Vec<Token>) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq + Eq,
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
        let prover_input = proof.io.input[0].clone();
        let prover_output = proof.io.output[0].clone();
        verify::<_, BasicTranscript<E>, _>(&self.verifier_ctx, proof.proof, proof.io)?;
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
    N: TensorTypeParam + Serialize + for<'b> Deserialize<'b>,
{
    fn observe<E: ExtensionField>(&self, step: usize, trace: &Trace<E, N>) {
        let tensor = trace
            .output
            .last()
            .unwrap()
            .tensor()
            .expect("hydration failed");
        let output_tokens_len = tensor.get_data().len();
        let input_tokens = self.tokenizer.tokenize(&self.input);
        let new_token = if output_tokens_len == 1 {
            *tensor.get_data().last().expect("last token must exist")
        } else {
            tensor.get_data()[input_tokens.len() + step - 1]
        };

        // let new_token = tensor.get_data().last().unwrap();
        let new_token = Token::from(Number::to_usize(&new_token));
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
        Number, init_test_logging,
        iop::chunking::LLMChunkingStrategy,
        model::llm::{
            Driver, LLMProverContext, LLMTokenizerObserver, LLMVerifierContext, WithMaxContext,
        },
        parser::{
            file_cache,
            gguf::{RawGGUF, TensorLoader},
            llm::{
                HFTokenizer, LLMTokenizer, Token,
                models::{
                    gemma3::{Gemma3, tests::GEMMA3_Q8},
                    gpt2::{GPT2, GPT2_Q8_0},
                },
                tokenizer::TokenizerLoader,
            },
        },
        rng_from_env_or_random,
        testing::Pcs,
    };
    use anyhow::Context;
    use ark_std::rand::Rng;
    use ff_ext::GoldilocksExt2;
    use tenstore::GenStore;
    use tracing::info;

    fn test_llm_driver_generic_prove_gpt2<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        const MAX_CONTEXT: usize = 10;
        init_test_logging("debug");

        // Load the model file
        // let model_path = std::path::PathBuf::from("assets/scripts/llms/toy_gpt2.gguf");
        // Load the model file
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GPT2_Q8_0}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        #[allow(clippy::type_complexity)]
        let (driver, prover_ctx, verifier_ctx): (
            _,
            LLMProverContext<GoldilocksExt2, Pcs<GoldilocksExt2>>,
            LLMVerifierContext<GoldilocksExt2, Pcs<GoldilocksExt2>>,
        ) = file_cache::deserialize_or_create_with(&cache_filename, || {
            let driver = Driver::load_from_model(
                GPT2,
                &RawGGUF::new(model_path.clone()),
                Some(MAX_CONTEXT),
            )?
            .into_provable_llm(None)?;

            let (prover_ctx, verifier_ctx) = driver
                .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
                .with_max_context(MAX_CONTEXT);

            Ok((driver, prover_ctx, verifier_ctx))
        })?;
        driver.model.describe();
        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = GPT2.load_tokenizer(&RawGGUF::new(model_path))?;
        let user_tokens = tokenizer.tokenize(sentence);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            &mut store,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1) // equivalent to sequential proving
        };
        let proof = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            LLMChunkingStrategy::default(),
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens)?;
        Ok(())
    }

    #[test]
    fn test_llm_driver_distributed_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<true>()
    }

    #[test]
    #[ignore = "Sequential case covered already by distributed prove test, use this test only when checking performance to ensure sequential proving is used"]
    fn test_llm_driver_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<false>()
    }

    #[test]
    fn test_llm_driver_inference() -> anyhow::Result<()> {
        init_test_logging("debug");
        const PRUNED_GPT2: &str = "gpt2.Q2_K.gguf";
        let model_path = file_cache::from_cache(PRUNED_GPT2)?;

        // let model_path = "assets/scripts/llms/toy_gpt2.gguf";
        let driver = Driver::load_from_model(GPT2, &RawGGUF::new(model_path.clone()), Some(10))?
            .into_provable_llm(None)?;
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = GPT2.load_tokenizer(&RawGGUF::new(model_path.clone()))?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        let mut store = GenStore::default();
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {user_tokens:?}");
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens,
            &mut store,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output);
        println!("{}", output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let driver = Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?;

        println!("LLM DRIVER: config: {:?}", driver.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let trace = driver.run::<GoldilocksExt2>(
            &user_tokens,
            &mut store,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_int_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let driver = Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?
            .into_provable_llm(None)?;

        println!("LLM DRIVER: config: {:?}", driver.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens,
            &mut store,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .get_data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    fn test_generic_prove_llm_gemma3<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
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
        #[allow(clippy::type_complexity)]
        let (driver, prover_ctx, verifier_ctx): (
            _,
            LLMProverContext<GoldilocksExt2, Pcs<GoldilocksExt2>>,
            LLMVerifierContext<GoldilocksExt2, Pcs<GoldilocksExt2>>,
        ) = file_cache::deserialize_or_create_with(&cache_filename, || {
            let gguf = RawGGUF::new(model_path.clone());
            let driver = Driver::load_from_model(Gemma3::new(), &gguf, Some(MAX_CONTEXT))?
                .into_provable_llm(None)?;

            let (prover_ctx, verifier_ctx) = driver
                .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
                .with_max_context(MAX_CONTEXT);

            Ok((driver, prover_ctx, verifier_ctx))
        })?;

        println!("LLM DRIVER: config: {:?}", driver.config);
        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let loader = TensorLoader::from_path(model_path)?;
        let tokenizer = HFTokenizer::sentencepiece_from_gguf(&loader)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            &mut store,
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;

        // Prove the trace
        let num_provers = Some(rng_from_env_or_random().gen_range(1..6));
        let proof = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            LLMChunkingStrategy::default(),
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens)?;
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_distribute_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<true>()
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<false>()
    }
}
