use std::{iter::once, ops::Deref};

use anyhow::{Result, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Claim, Element, NextPowerOfTwo, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            Splittable, VerifiableCtx,
        },
    },
    model::Step,
    padding::{PaddingMode, ShapeData, ShapeInfo},
    parser::{gguf, json},
    poly_commit::identity_eval,
    quantization::{self, Quantize, ToElement, ToField},
    tensor::{CommitmentId, TensorHandle, TensorTypeParam, WrappedTensor},
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};

/// The short name used to identify the embeddings layer
pub const EMBEDDINGS_LAYER: &str = "EMBD";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct Embeddings<T>
where
    T: TensorTypeParam,
{
    pub(crate) mat: TensorHandle<T>,
    pub(crate) emb_size: usize,
    pub(crate) vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsCtx {
    vocab_size: usize,
    emb_size: usize,
    embedding_key: CommitmentId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct EmbeddingsProof<F: PrimeField> {
    /// the actual sumcheck proof proving the matmul protocol
    pub(crate) sumcheck: IOPProof<F>,
    /// The individual evaluations of the individual polynomial for the last random part of the
    /// sumcheck. One for each polynomial involved in the "virtual poly".
    /// Since we only support quadratic right now it's a flat list.
    #[serde(with = "dp_crypto::serialization")]
    individual_claims: Vec<F>,
}

impl<N> Embeddings<N>
where
    N: TensorTypeParam,
{
    pub fn new(emb: TensorHandle<N>) -> anyhow::Result<Self> {
        let emb_size = emb.shape()[1];
        let vocab_size = emb.shape()[0];

        Ok(Self {
            mat: emb.wrapped_tensor_variant()?,
            emb_size,
            vocab_size,
        })
    }

    /// Split the point over which the 2d output tensor is evaluated into 2 sub-points:
    /// - The first sub-point refers to the row variables of the output tensor
    /// - The second sub-point refers to the column variables of the output tensor
    fn split_output_point<F: PrimeField>(
        last_claim: &Claim<F>,
        emb_size: usize,
    ) -> anyhow::Result<(&[F], &[F])> {
        let num_vars = emb_size.next_power_of_two().ilog2() as usize;
        // column variables are the least significant ones
        let column_point = &last_claim.point[..num_vars];
        // row variables are the most significant ones
        let row_point = &last_claim.point[num_vars..];

        Ok((row_point, column_point))
    }

    /// Build points over which the evaluations produced by the sum-check in proving protocol are evaluated.
    /// - The first point being returned is the point for the claim about the one-hot encoded input
    /// - The second point being returned is the point for the claim about the emebdding matrix
    fn build_points_for_claims<F: PrimeField>(
        last_claim: &Claim<F>,
        emb_size: usize,
        sumcheck_point: &[F],
    ) -> anyhow::Result<(Vec<F>, Vec<F>)> {
        let (row_point, column_point) = Self::split_output_point(last_claim, emb_size)?;
        let one_hot_claim_point = sumcheck_point
            .iter()
            .chain(row_point)
            .cloned()
            .collect_vec();
        let embedding_mat_point = column_point
            .iter()
            .chain(sumcheck_point.iter())
            .cloned()
            .collect_vec();
        Ok((one_hot_claim_point, embedding_mat_point))
    }
}

fn output_shapes(
    input_shapes: &[Shape],
    padding_mode: PaddingMode,
    embedding_size: usize,
) -> Vec<Shape> {
    assert!(
        input_shapes.len() == 1,
        "embeddings only support 1 input tensor"
    );
    assert_eq!(
        input_shapes[0].rank(),
        1,
        "embeddings only support 1d tensors"
    );
    let seq_len = input_shapes[0].dim(0);
    let shape = match padding_mode {
        PaddingMode::NoPadding => Shape::new(vec![seq_len, embedding_size]),
        PaddingMode::Padding => Shape::new(vec![seq_len, embedding_size]).next_power_of_two(),
    };
    vec![shape]
}

impl<N: TensorTypeParam> OpInfo for Embeddings<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        ensure!(
            input_shapes.len() == 1,
            "embeddings only support 1 input tensor, got {}",
            input_shapes.len()
        );
        ensure!(
            input_shapes[0].rank() == 1,
            "embeddings only support 1d tensors, got {}",
            input_shapes[0].rank(),
        );
        let seq_len = input_shapes[0].dim(0);
        let shape = match padding_mode {
            PaddingMode::NoPadding => Shape::new(vec![seq_len, self.emb_size]),
            PaddingMode::Padding => Shape::new(vec![seq_len, self.emb_size]).next_power_of_two(),
        };
        Ok(vec![shape])
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(1)
    }

    fn describe(&self) -> String {
        format!(
            "Embeddings(vocab:{:?}, hidden:{:?})",
            self.vocab_size, self.emb_size
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for Embeddings<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> anyhow::Result<LayerOut<f32>> {
        ensure!(
            inputs.iter().all(|x| { x.rank() == 1 }),
            "embeddings only support 1d tensors: {:?}",
            inputs.iter().map(|x| x.shape()).collect::<Vec<_>>()
        );
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");

        let input = inputs[0];

        let weights = self.mat.wrapped_tensor()?;
        let res = weights.clone().select(0, input.clone().int())?;

        Ok(LayerOut::from_tensor(res))
    }
}

impl Evaluate<Element> for Embeddings<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> anyhow::Result<LayerOut<Element>> {
        ensure!(
            inputs.iter().all(|x| { x.rank() == 1 }),
            "embeddings only support 1d tensors: {:?}",
            inputs.iter().map(|x| x.shape()).collect::<Vec<_>>()
        );
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        let input = inputs[0];

        // we still uses this evaluation for inference as it's quicker
        // than doing the matmul with one hot encoding. Proving however will generate
        // the one hot encoding and do the matmul.

        let weights = self.mat.wrapped_tensor()?;

        weights
            .clone()
            .select(0, input.clone())
            .map(LayerOut::from_tensor)
    }
}

impl PadOp for Embeddings<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        ensure!(
            si.shapes.len() == 1,
            "embeddings only support 1 input tensor"
        );
        // we need to give the shapes that the one hot encoding will have
        let shape_data = si.shapes.first().unwrap();
        ensure!(
            shape_data.input_shape_og.rank() == 1,
            "embeddings only support 1d tensors"
        );
        // Need the unpadded_output_shape
        let emb_size = self.mat.unpadded_shape().dim(-1);
        let output_unpadded_shape = Shape::new(vec![shape_data.input_shape_og[0], emb_size]);
        let padded_output_shape = output_unpadded_shape.next_power_of_two();

        si.shapes = vec![ShapeData {
            input_shape_og: output_unpadded_shape,
            input_shape_padded: padded_output_shape,
            ignore_garbage_pad: None,
        }];

        let padded_emb = self.mat;

        Ok(Self {
            mat: padded_emb,
            ..self
        })
    }
}

impl ProveInfo for Embeddings<Element> {
    fn step_info<F: PrimeField>(
        &self,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        let tensor = self.mat.wrapped_tensor()?;
        let data = Vec::from(tensor.clone().pad_next_power_of_two());

        aux.last_output_shape = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;
        aux.model_polys = Some(once((CommitmentId::from(self.mat.storage_key()), data)).collect());

        Ok((
            LayerCtx::Embeddings(EmbeddingsCtx {
                vocab_size: self.vocab_size,
                emb_size: self.emb_size,
                embedding_key: CommitmentId::from(self.mat.storage_key()),
            }),
            aux,
        ))
    }
}

impl OpInfo for EmbeddingsCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(output_shapes(input_shapes, padding_mode, self.emb_size))
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(1)
    }

    fn describe(&self) -> String {
        format!(
            "EmbeddingsCtx(vocab:{:?}, hidden:{:?})",
            self.vocab_size, self.emb_size
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Splittable for EmbeddingsCtx {}

impl QuantizeOp for Embeddings<f32> {
    type QuantizedOp = Embeddings<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        _input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        _output_scaling: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let Embeddings {
            mat,
            emb_size,
            vocab_size,
        } = self;

        // We pick the domain size to be the largest multiple of quantization::BIT_LEN that is less than or equal to 24.
        let mut multiple = 1usize;
        while (multiple + 1) * *quantization::BIT_LEN <= 24 {
            multiple += 1;
        }
        let multiple_bit_size = multiple * *quantization::BIT_LEN;

        let shift = multiple_bit_size - 1;
        let embeddings_domain_min: Element = -1 << shift;
        let embeddings_domain_max: Element = (1 << shift) - 1;
        let scale_factor = ScalingFactor::from_absolute_max(
            mat.max_abs()?,
            Some((embeddings_domain_min, embeddings_domain_max)),
        );
        let quantised_mat = mat.quantize(&scale_factor);
        let qemb = Embeddings {
            mat: quantised_mat,
            emb_size,
            vocab_size,
        };
        Ok(QuantizeOutput::new(qemb, vec![scale_factor]))
    }
}

impl<F, PCS> ProvableOp<F, PCS> for Embeddings<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = EmbeddingsCtx;

    fn prove<T: Transcript>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        // we first construct the one hot encoding from the input indices and then we run
        // the matmul protocol.
        ensure!(
            step_data.node_inputs.len() == 1,
            "embeddings only support 1 input tensor"
        );
        ensure!(
            last_claims.len() == 1,
            "embeddings only support 1 last claim"
        );
        let last_claim = last_claims[0];
        let input_tensors = step_data.padded_input_tensors()?;
        let input = &input_tensors[0];
        let unpadded_length = input.unpadded_shape().numel();

        let (row_point, column_point) = Self::split_output_point(last_claim, self.emb_size)?;

        // we need to compute the vector `reduced_one_hot` whose MLE corresponds to the MLE of the one-hot
        // encoded input matrix, with row variables fixed to `row_point`.
        // Relying on the sparse structure of the one-hot encoded input matrix, this vector can be computed
        // as `reduced_one_hot[x[i]] += \beta(i, row_point)`, for all items `x[i]` in the input tensor

        // we precompute all items `\beta(i, row_point)` for all `i` between `0` and `x.len()`
        let beta_vec = evals(row_point);
        let vocab_size = self.vocab_size.next_power_of_two();
        let emb_size = self.emb_size.next_power_of_two();
        // we now build the `reduced_one_hot` vector as `reduced_one_hot[x[i]] += beta_vec[i]`
        let mut reduced_one_hot = vec![F::ZERO; vocab_size];

        input
            .data()
            .iter()
            .take(unpadded_length)
            .enumerate()
            .for_each(|(i, &x)| {
                reduced_one_hot[x as usize] += beta_vec[i];
            });

        let reduced_one_hot = Tensor::new(vec![1, vocab_size].into(), reduced_one_hot)?;

        let embedding_matrix = Tensor::try_from(&self.mat)?.pad_next_power_of_two();

        ensure!(
            vocab_size == embedding_matrix.nrows_2d()?,
            "Expected {vocab_size} rows for embedding matrix, found {}",
            embedding_matrix.nrows_2d()?,
        );

        ensure!(
            emb_size == embedding_matrix.ncols_2d()?,
            "Expected {emb_size} columns for embedding matrix, found {}",
            embedding_matrix.ncols_2d()?,
        );

        let input_mle = reduced_one_hot.to_mle_2d()?;

        let mut embedding_mat_mle = embedding_matrix.to_2d_mle()?;

        embedding_mat_mle.fix_low_variables_in_place_parallel(column_point);

        // check that after fixing the variables in both matrices the number of free
        // variables is the same
        assert_eq!(input_mle.num_vars(), embedding_mat_mle.num_vars());

        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let embedding_mat_expr = expr_builder.lift(Either::Left(&embedding_mat_mle));
        let virtual_poly = expr_builder.to_virtual_polys(&[input_expr * embedding_mat_expr], &[]);
        let (proof, state) = IOPProverState::<F>::prove(virtual_poly, prover.transcript);

        // sum-check will produce claims about `reduced_one_hot` vector and the `embedding_matrix`. We need
        // to commit build a claim for the one-hot encoded input tensor from the first claim, and produce an
        // opening proof for the second claim
        let final_evaluations = state.get_mle_flatten_final_evaluations();
        let one_hot_eval = final_evaluations[0];
        let embedding_mat_eval = final_evaluations[1];

        let (one_hot_claim_point, embedding_mat_point) = Self::build_points_for_claims(
            last_claim,
            self.emb_size,
            &state.collect_raw_challenges(),
        )?;

        let output_claim = Claim::new(one_hot_claim_point, one_hot_eval);

        let embedding_claim = Claim::new(embedding_mat_point, embedding_mat_eval);

        prover.add_common_claims(
            node_id,
            once((CommitmentId::from(self.mat.storage_key()), embedding_claim)).collect(),
        );

        prover.push_proof(
            node_id,
            LayerProof::Embeddings(EmbeddingsProof {
                sumcheck: proof,
                individual_claims: final_evaluations,
            }),
        );
        Ok(vec![output_claim])
    }
}

impl<F, PCS> VerifiableCtx<F, PCS> for EmbeddingsCtx
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = EmbeddingsProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        _shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        ensure!(
            last_claims.len() == 1,
            "embeddings only support 1 last claim"
        );
        let last_claim = &last_claims[0];
        let num_vars = self.vocab_size.next_power_of_two().ilog2() as usize;
        let subclaim = IOPVerifierState::<F>::verify(
            last_claim.eval,
            &proof.sumcheck,
            &from_mle_list_dimensions(&[vec![num_vars, num_vars]]),
            verifier.transcript,
        );

        // build claims produced by sum-check: a claim about the one-hot encoded input, and a claim about
        // the embedding matrix

        let (one_hot_claim_point, emebdding_mat_point) =
            Embeddings::<Element>::build_points_for_claims(
                last_claim,
                self.emb_size,
                &subclaim.point,
            )?;
        let one_hot_eval = proof.individual_claims[0];
        let embedding_mat_eval = proof.individual_claims[1];

        let one_hot_claim = Claim::new(one_hot_claim_point, one_hot_eval);

        let embedding_mat_claim = Claim::new(emebdding_mat_point, embedding_mat_eval);

        verifier.add_common_claims(
            node_id,
            once((self.embedding_key.clone(), embedding_mat_claim)).collect(),
        );

        // SUMCHECK verification part
        // Instead of computing the polynomial at the random point requested like this
        // let computed_point = vp.evaluate(
        //     subclaim
        //         .point
        //         .iter()
        //         .map(|c| c.elements)
        //         .collect_vec()
        //         .as_ref(),
        //
        // We compute the evaluation directly from the individual final evaluations of each polynomial
        // involved in the sumcheck the prover's giving,e.g. y(res) = SUM f_i(res)
        ensure!(
            one_hot_eval * embedding_mat_eval == subclaim.expected_evaluation,
            "sumcheck claim failed in Logits layer verifier",
        );

        // the first claim is the one hot encoding claim. To verify it we need to
        // efficiently evaluate the one hot encoding on it - we do this "at the end" of the verification
        // procedure to respect the framework's order of operations. The logic is in `verify_input_claim`.
        Ok(vec![one_hot_claim])
    }

    fn verify_input_claim<D: Deref<Target = F> + ToField<F>, A: AsRef<Tensor<D>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<F>],
    ) -> anyhow::Result<()> {
        // TODO verify efficiently the one hot encoding claim
        ensure!(inputs.len() == 1, "embeddings only support 1 input tensor");
        ensure!(claims.len() == 1, "embeddings only support 1 claim");
        let input = inputs[0].as_ref();
        let unpadded_length = input.unpadded_shape().numel();
        let one_hot_claim = &claims[0];
        let vocab_nv = self.vocab_size.next_power_of_two().ilog2();
        let seq_len_nv = input.shape().dim(0).next_power_of_two().ilog2();
        ensure!(
            vocab_nv + seq_len_nv == one_hot_claim.point.len() as u32,
            "vocab_nv: {vocab_nv}, seq_len_nv: {seq_len_nv}, one_hot_claim.point.len(): {}",
            one_hot_claim.point.len()
        );
        let (r1, r2) = one_hot_claim.point.split_at(vocab_nv as usize);
        let b1 = evals(r2);
        let sum = input.data().iter().take(unpadded_length).zip(b1).fold(
            F::ZERO,
            |sum, (token, beta)| {
                let token_value = token.to_element() as usize;
                let token_le_bits = to_bit_sequence_le(token_value, r1.len())
                    .map(|b| F::from(b as u64))
                    .collect_vec();
                let selector = beta * identity_eval(r1, &token_le_bits);
                sum + selector
            },
        );
        ensure!(
            sum == one_hot_claim.eval,
            "one hot encoding claim is incorrect"
        );
        Ok(())
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        // No commitment so just return Ok(())
        Ok(())
    }
}

impl Embeddings<f32> {
    pub fn from_json(l: &json::FileTensorLoader) -> anyhow::Result<Self> {
        let emb_tensor = l.get_tensor("token_embd.weight")?;
        Embeddings::new(emb_tensor.into())
    }
    pub fn from_loader(loader: &gguf::FileTensorLoader) -> anyhow::Result<Self> {
        let emb_tensor = loader.get_tensor("token_embd.weight")?;
        Embeddings::new(emb_tensor.into())
    }
    pub fn from_safetensors_loader(
        loader: &crate::parser::safe::FileTensorLoader,
    ) -> anyhow::Result<Self> {
        // Try common key names across models; prefer Gemma3 safetensors first
        let emb_tensor = loader
            .get_tensor("model.embed_tokens.weight")
            .or_else(|_| loader.get_tensor("token_embd.weight"))
            .or_else(|_| loader.get_tensor("tok_embeddings.weight"))
            .or_else(|_| loader.get_tensor("wte.weight"))?;
        Embeddings::new(emb_tensor.into())
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::rand::Rng;
    use proptest::prelude::*;
    use std::{fmt::Debug, ops::Range};
    use tenstore::{GenStore, StorageKey};

    use crate::{
        Element, Number,
        layers::Layer,
        model::{
            Model,
            test::{prove_quantized_model, quantize_model},
        },
        quantization::{Quantize, ToField},
        rng_from_env_or_random,
    };

    use super::*;

    type F = ark_bn254::Fr;

    fn generate_unique_random_indices(seq_len: usize, vocab_size: usize) -> Vec<usize> {
        let mut ctr = 0;
        let mut rng = rng_from_env_or_random();
        while ctr < 10 {
            let d = (0..seq_len)
                .map(|_| rng.gen_range(0..vocab_size))
                .collect::<Vec<_>>();
            let mut dd = d.clone();
            dd.sort();
            dd.dedup();
            if dd.len() == seq_len {
                return d;
            }
            ctr += 1;
        }
        panic!("failed to generate unique random indices");
    }

    #[test]
    fn test_one_hot_encoding_proving() -> anyhow::Result<()> {
        let seq_len: usize = 5;
        let vocab_size: usize = 200;
        let emb_size: usize = 10;
        let indices = (0..seq_len)
            .map(|_| rng_from_env_or_random().gen_range(0..vocab_size) as f32)
            .collect::<Vec<_>>();
        let input_shape = Shape::from(vec![seq_len]);
        let input = Tensor::new(input_shape.clone(), indices.clone()).unwrap();
        let mut model = Model::new_from_input_shapes(vec![input_shape.clone()]);

        let embeddings_value = Tensor::random(&Shape::new(vec![vocab_size, emb_size]));
        let embeddings = Embeddings::new(TensorHandle::from_tensor(
            StorageKey::from("embeddings_mat"),
            GenStore::new_empty(),
            embeddings_value.clone(),
        ))?;
        let _ = model
            .add_consecutive_layer(Layer::Embeddings(embeddings), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        model.describe();

        let mut store = GenStore::default();
        let (quantized_model, _) = quantize_model(
            model,
            vec![input.clone()],
            Some(vec![input.clone()]),
            &mut store,
        )?;

        let quantised_input_data = input
            .data()
            .iter()
            .map(|val| *val as Element)
            .collect::<Vec<Element>>();
        let quantised_input = Tensor::new(input.shape().clone(), quantised_input_data)?;

        prove_quantized_model(quantized_model, vec![quantised_input], &mut store)?;

        Ok(())
    }

    fn one_hot_encoding<F: PrimeField>(indices: &[F], vocab_size: usize) -> Tensor<F> {
        let mut data = Vec::new();
        for idx in indices {
            let mut one_hot = vec![F::ZERO; vocab_size];
            let idx: usize = idx.to_element().try_into().unwrap();
            one_hot[idx] = F::ONE;
            data.extend_from_slice(&one_hot);
        }
        Tensor::new(vec![indices.len(), vocab_size].into(), data).unwrap()
    }

    #[test]
    fn test_one_hot_encoding_inference() -> anyhow::Result<()> {
        let seq_len: usize = 5;
        let indices_elem: Vec<Element> = (0..seq_len).map(|i| i as Element).collect::<Vec<_>>();
        let indices: Tensor<F> =
            Tensor::<Element>::new(vec![5].into(), indices_elem.clone())?.to_field();
        let vocab_size = 6;
        let emb_size = 10;
        let one_hot = one_hot_encoding(indices.data(), vocab_size);
        let expected_shape: Shape = vec![indices.shape().numel(), vocab_size].into();
        assert_eq!(*one_hot.shape(), expected_shape);
        assert_eq!(
            one_hot.data(),
            &[
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ONE,
                F::ZERO,
            ]
        );

        let emb = Tensor::<Element>::random(&vec![vocab_size, 10].into());
        let embeddings = Embeddings::new(TensorHandle::from_tensor(
            StorageKey::from("embeddings_mat"),
            GenStore::new_empty(),
            emb.clone(),
        ))?;
        let input = Tensor::new(vec![seq_len].into(), indices_elem.clone()).unwrap();
        let out = embeddings.evaluate(&[&input.as_wrapped()])?;
        let expected_shape = Shape::new(vec![seq_len, emb_size]);
        assert_eq!(Shape::from(out.outputs()[0].shape()), expected_shape);
        let onehot_result = one_hot.matmul(&emb.to_field())?;

        assert_eq!(onehot_result, out.outputs()[0].to_native().to_field());

        Ok(())
    }

    #[test]
    fn test_embeddings() -> anyhow::Result<()> {
        let seq_len = 10;
        let vocab_size = 100;
        let emb_size = 20;
        // generate the vector of embeddings for a given index
        let emb_vector = |idx: usize| -> Vec<Element> {
            (0..emb_size)
                .map(|j| Element::from((10000 * idx + j) as Element))
                .collect()
        };
        let table = (0..vocab_size).flat_map(emb_vector).collect::<Vec<_>>();
        let emb_tensor = Tensor::new(vec![vocab_size, emb_size].into(), table)?;
        let embeddings = Embeddings::new(TensorHandle::from_tensor(
            StorageKey::from("embeddings_mat"),
            GenStore::new_empty(),
            emb_tensor,
        ))?;

        // generate random indices
        let input_data = generate_unique_random_indices(seq_len, vocab_size)
            .into_iter()
            .map(|x| Element::from(x as Element))
            .collect::<Vec<_>>();
        let x = Tensor::new(vec![seq_len].into(), input_data.clone()).unwrap();
        let out = embeddings.evaluate(&[&x.as_wrapped()])?;
        assert_eq!(out.outputs()[0].shape(), vec![seq_len, emb_size].into());
        // for each input index, check that the embedding vector is the correct one
        for (idx, table_idx) in input_data.iter().enumerate() {
            let emb = emb_vector(*table_idx as usize);
            let out_emb =
                out.outputs()[0].get_data()[idx * emb_size..(idx + 1) * emb_size].to_vec();
            assert_eq!(emb, out_emb);
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn test_embeddings_with_f32(input in any_input::<f32>(1..256, 1..8, 1..32)) {
            let Input { emb, input } = input;
            let seq_len = input.shape()[0];
            let vocab_size = emb.shape()[0];
            let emb_size = emb.shape()[1];
            let emb_data = emb.data();

            let new_emb = input
                .data()
                .iter()
                .flat_map(|v| {
                    let idx = v.to_usize();
                    assert!(
                        idx < vocab_size,
                        "idx {idx} out of bounds for vocab size {vocab_size}"
                    );
                    let emd_idx = idx * emb_size;
                    emb_data[emd_idx..emd_idx + emb_size].to_vec()
                })
                .collect::<Vec<_>>();
            let out_shape = Shape::new(vec![seq_len, emb_size]);
            let expected = Tensor::new(out_shape, new_emb).unwrap();

            let layer = Embeddings::new(TensorHandle::from_tensor(
                StorageKey::from("embeddings_mat"),
                GenStore::new_empty(),
                emb,
            )).unwrap();
            let computed = layer.evaluate(&[&input.as_wrapped()]).unwrap();

            prop_assert_eq!(expected, computed.outputs[0].to_native());
        }

        #[test]
        fn test_embeddings_with_element(input in any_input::<Element>(1..256, 1..8, 1..32)) {
            let Input { emb, input } = input;
            let seq_len = input.shape()[0];
            let vocab_size = emb.shape()[0];
            let emb_size = emb.shape()[1];

            let scaling = ScalingFactor::from_span(
                <f32 as Number>::MIN,
                <f32 as Number>::MAX,
                Some((0, vocab_size as Element)),
            );
            let input = input.quantize(&scaling);

            let emb_data = emb.data();
            let new_emb = input
                .data()
                .iter()
                .flat_map(|v| {
                    let idx = v.to_usize();
                    assert!(
                        idx < vocab_size,
                        "idx {idx} out of bounds for vocab size {vocab_size}"
                    );
                    let emd_idx = idx * emb_size;
                    emb_data[emd_idx..emd_idx + emb_size].to_vec()
                })
                .collect::<Vec<_>>();
            let out_shape = Shape::new(vec![seq_len, emb_size]);
            let expected = Tensor::new(out_shape, new_emb).unwrap();

            let layer = Embeddings::new(TensorHandle::from_tensor(
                StorageKey::from("embeddings_mat"),
                GenStore::new_empty(),
                emb,
            )).unwrap();
            let computed = layer.evaluate(&[&input.as_wrapped()]).unwrap();

            prop_assert_eq!(expected, computed.outputs[0].to_native());
        }
    }

    struct Input<T> {
        emb: Tensor<T>,
        input: Tensor<f32>,
    }

    impl<T> Debug for Input<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input").finish_non_exhaustive()
        }
    }

    fn any_input<T: TensorTypeParam>(
        vocab_size: Range<usize>,
        emb_size: Range<usize>,
        input_size: Range<usize>,
    ) -> impl Strategy<Value = Input<T>> {
        (vocab_size, emb_size, input_size).prop_flat_map(|(vocab_size, emb_size, input_size)| {
            let shape = Shape::new(vec![vocab_size, emb_size]);
            let emb = Tensor::<T>::any(shape.clone());
            let input = Tensor::<f32>::any(Shape::new(vec![emb_size * input_size]));
            (emb, input).prop_map(|(emb, input)| Input { emb, input })
        })
    }
}
