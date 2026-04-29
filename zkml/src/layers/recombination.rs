use std::iter::once;

use anyhow::ensure;
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::eq::evals,
    util::ceil_log2,
};
use itertools::{Itertools, izip};
use serde::{Deserialize, Serialize};

use crate::{
    Claim, Element, InitTranscript, NextPowerOfTwo, Prover, Shape,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
        split::SplitLayer,
    },
    model::Step,
    padding::{PaddingMode, ShapeInfo},
    tensor::{TensorTypeParam, WrappedTensor},
    try_unzip,
};

pub(crate) const RECOMBINATION_LAYER: &str = "RECOMBINATION";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct RecombinationProof<F: PrimeField> {
    #[serde(with = "dp_crypto::serialization")]
    evals: Vec<Vec<F>>,
}

/// A layer that recombines the given number of chunks for each input tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecombinationLayer {
    pub(crate) num_chunks: Vec<usize>,
}

impl<'a> From<&'a SplitLayer> for RecombinationLayer {
    fn from(value: &'a SplitLayer) -> Self {
        Self {
            num_chunks: value.num_chunks.clone(),
        }
    }
}

impl RecombinationLayer {
    pub fn new(num_chunks: Vec<usize>) -> Self {
        Self { num_chunks }
    }

    /// Split the point provided as input in 2 set of coordinates:
    /// - the first set is the point where to evaluate claims about the chunks
    /// - the second set are the coordinates relative to the number of chunks
    fn split_point_for_chunk<F: PrimeField>(num_chunks: usize, point: &[F]) -> (&[F], &[F]) {
        let num_chunks_vars = ceil_log2(num_chunks);
        // discard the last `num_chunks_vars` coordinates
        point.split_at(point.len() - num_chunks_vars)
    }
}

impl OpInfo for RecombinationLayer {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> anyhow::Result<Vec<Shape>> {
        let recombined_output_shapes = self
            .num_chunks
            .iter()
            .scan(0, |acc, &num_chunks| {
                let shapes = &input_shapes[*acc..*acc + num_chunks];
                *acc += num_chunks;
                Some(shapes)
            })
            .map(|shapes| {
                ensure!(shapes.iter().map(|shape| shape[1..].to_vec()).all_equal());
                // compute the shape given by concatenating the first dimention of all shapes
                let first_dimension = shapes.iter().map(|shape| shape[0]).sum();
                Ok(Shape::from_it(
                    once(first_dimension).chain(shapes[0][1..].to_vec()),
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(if let PaddingMode::Padding = padding_mode {
            // pad the shapes
            recombined_output_shapes
                .into_iter()
                .map(|shape| shape.next_power_of_two())
                .collect()
        } else {
            recombined_output_shapes
        })
    }

    fn num_outputs(&self, num_inputs: usize) -> anyhow::Result<usize> {
        ensure!(num_inputs == self.num_chunks.iter().sum::<usize>());
        Ok(self.num_chunks.len())
    }

    fn describe(&self) -> String {
        format!("Recombine(num_chunks={:?})", self.num_chunks)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<T: TensorTypeParam> Evaluate<T> for RecombinationLayer {
    fn evaluate(&self, inputs: &[&WrappedTensor<T>]) -> anyhow::Result<LayerOut<T>> {
        let outputs = self
            .num_chunks
            .iter()
            .scan(0, |acc, &num_chunks| {
                let chunks = inputs[*acc..*acc + num_chunks].to_vec();
                *acc += num_chunks;
                Some(chunks)
            })
            .map(|chunks| WrappedTensor::cat(chunks.into_iter().cloned().collect(), 0))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(LayerOut::from_vec(outputs))
    }
}

impl PadOp for RecombinationLayer {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        si.update_shapes(&self)?;
        Ok(self)
    }
}

impl ProveInfo for RecombinationLayer {
    fn step_info<F: PrimeField>(
        &self,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        let output_shapes = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;
        aux.last_output_shape = output_shapes;
        let layer_ctx = LayerCtx::Recombination(self.clone());
        Ok((layer_ctx, aux))
    }
}

impl<F, PCS> ProvableOp<F, PCS> for RecombinationLayer
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = RecombinationLayer;

    fn prove<'a, 'b, 'c, 'd, T: Transcript + InitTranscript>(
        &'a self,
        node_id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        let inputs = step_data.padded_input_tensors()?;
        let (input_claims, evals): (Vec<Vec<Claim<F>>>, Vec<Vec<F>>) = try_unzip(
            self.num_chunks
                .iter()
                .scan(0, |acc, &num_chunks| {
                    let chunks = &inputs[*acc..*acc + num_chunks];
                    *acc += num_chunks;
                    Some(chunks)
                })
                .zip(last_claims)
                .map(|(chunks, claim)| {
                    let num_chunks = chunks.len();
                    let eval_point = Self::split_point_for_chunk(num_chunks, &claim.point).0;
                    // evaluate each chunk MLE on this point
                    try_unzip(chunks.iter().map(|t| {
                        let chunk_mle = t.to_field_mle();
                        let chunk_point = &eval_point[..chunk_mle.num_vars()];
                        let eval = chunk_mle.evaluate(chunk_point)?;
                        anyhow::Ok((Claim::new(chunk_point.to_vec(), eval), eval))
                    }))
                }),
        )?;

        prover.push_proof(
            node_id,
            LayerProof::Recombination(RecombinationProof { evals }),
        );

        Ok(input_claims.concat())
    }
}

impl<F, PCS> VerifiableCtx<F, PCS> for RecombinationLayer
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    type Proof = RecombinationProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        _verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        _node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        // the verifier basically needs to check that the claims about the inputs, which are found in the proof,
        // are consistent with respect to the output claims

        let unpadded_input_shapes = &shape_step.unpadded_input_shape;
        let input_claims = self.num_chunks.iter()
        .scan(0, |acc, &num_chunks| {
            let chunk_num_vars = unpadded_input_shapes[*acc..*acc + num_chunks].iter()
                .map(|shape| shape.next_power_of_two().num_vars().into_iter().sum())
                .collect_vec();
            *acc += num_chunks;
            Some((chunk_num_vars, num_chunks))
        })
        .enumerate().map(|(i, (chunk_num_vars, num_chunks)) | {
            let chunk_evals = &proof.evals[i];
            ensure!(chunk_evals.len() == num_chunks, "Expected {num_chunks} evals in Recombination proof for {i}-th output, found {}", chunk_evals.len());
            let output_claim = last_claims[i];
            let (chunk_point, num_chunk_coords) = Self::split_point_for_chunk(num_chunks, output_claim.point());
            let beta_evals = evals(num_chunk_coords);
            let combined_eval = izip!(chunk_evals, beta_evals, &chunk_num_vars)
                .fold(F::ZERO, |acc, (&eval, beta_eval, &num_vars)| {
                    // we first need to adapt eval to the shape of the chunk
                    let (_, padding_point) = chunk_point.split_at(num_vars);
                    // we now need to add `padding_point` coordinates to take into account padding of
                    // the chunk to the size of the other padded chunks
                    let eval = padding_point
                        .iter()
                        .fold(eval, |eval, &padding_coord| eval * (F::ONE - padding_coord));
                    acc + eval * beta_eval
                });
            ensure!(
                combined_eval == output_claim.eval,
                "Mismatch in output claim evaluation when verifying Recombination layer: expected {}, got {combined_eval}", 
                output_claim.eval
            );
            Ok(chunk_evals.iter().zip(chunk_num_vars).map(|(&eval, num_vars)| {
                Claim::new(chunk_point[..num_vars].to_vec(), eval)
            }).collect_vec())
        }).collect::<anyhow::Result<Vec<_>>>()?;

        Ok(input_claims.concat())
    }

    fn write_proof_to_transcript<T: dp_crypto::arkyper::transcript::Transcript>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use itertools::Itertools;
    use tenstore::GenStore;

    use crate::{
        Element, Shape, Tensor,
        graph::NodeInput,
        init_test_logging,
        layers::{Layer, einsum::EinSum, recombination::RecombinationLayer, split::SplitLayer},
        model::{Model, test::prove_quantized_model},
        padding::pad_model,
        rng_from_env_or_random,
        tensor::KeyedTensor,
    };

    #[test]
    fn test_split_and_recombination() -> anyhow::Result<()> {
        let mut rng = rng_from_env_or_random();
        let input_shapes = vec![
            Shape::new(vec![
                rng.gen_range(34..100),
                rng.gen_range(10..31),
                rng.gen_range(10..31),
            ]),
            Shape::new(vec![rng.gen_range(34..100), rng.gen_range(10..31)]),
            Shape::new(vec![
                rng.gen_range(34..100),
                rng.gen_range(10..31),
                rng.gen_range(10..31),
            ]),
        ];

        let mut model = Model::<Element>::new_from_input_shapes(input_shapes);
        let max_num_chunks = 4;
        let unpadded_input_shapes = model.input_shapes();
        let split_layer =
            SplitLayer::new_from_input_shapes(max_num_chunks, &unpadded_input_shapes)?;
        let recombination_layer = RecombinationLayer::from(&split_layer);
        let split_id = model.add_consecutive_layer(Layer::Split(split_layer), None)?;

        model.add_consecutive_layer(Layer::Recombination(recombination_layer), Some(split_id))?;

        model.automatic_output_labelling()?;
        init_test_logging("debug");
        model.describe();

        let inputs = model
            .input_shapes()
            .iter()
            .map(Tensor::random)
            .collect_vec();

        let padded_inputs = inputs
            .iter()
            .map(|input| input.pad_next_power_of_two())
            .collect_vec();

        // run unpadded model
        let trace = model.run(inputs.clone(), &mut GenStore::default())?;

        trace
            .outputs()
            .iter()
            .zip(inputs)
            .try_for_each(|(out, input)| {
                assert_eq!(*out.tensor()?, input);
                anyhow::Ok(())
            })?;

        // now pad the model and run over padded inputs
        let padded_model = pad_model(model)?;

        let trace = padded_model.run(padded_inputs.clone(), &mut GenStore::default())?;

        trace
            .outputs()
            .iter()
            .zip(padded_inputs)
            .try_for_each(|(out, input)| {
                assert_eq!(out.tensor()?.pad_next_power_of_two(), input);
                Ok(())
            })
    }

    #[test]
    fn test_recombination_layer() -> anyhow::Result<()> {
        let mut rng = rng_from_env_or_random();
        let input_shapes = vec![
            Shape::new(vec![
                rng.gen_range(34..100),
                rng.gen_range(10..31),
                rng.gen_range(10..31),
            ]),
            Shape::new(vec![rng.gen_range(34..100), rng.gen_range(10..31)]),
            Shape::new(vec![
                rng.gen_range(34..100),
                rng.gen_range(10..31),
                rng.gen_range(10..31),
            ]),
        ];

        let first_input_last_dim = *input_shapes[0].last().unwrap();
        let weight = KeyedTensor::new(
            "weight",
            Tensor::random(&Shape::new(vec![
                first_input_last_dim,
                first_input_last_dim,
            ])),
        );
        let mut model = Model::<Element>::new_from_input_shapes(input_shapes);

        let einsum_id = model.graph_mut().add_inner(Layer::EinSum(EinSum::new(
            "A(ijk)@B(kl)->O(ijl)".into(),
            vec![Some(weight.into())],
            vec![None],
        )?))?;

        model.connect_model_input(0, NodeInput::new(einsum_id, 0))?;

        let unpadded_input_shapes = model.input_shapes();
        let max_num_chunks = 4usize;
        let split_layer =
            SplitLayer::new_from_input_shapes(max_num_chunks, &unpadded_input_shapes)?;
        let recombination_layer = RecombinationLayer::from(&split_layer);
        let split_id = model
            .graph_mut()
            .add_inner(Layer::Split(split_layer))
            .unwrap();

        // route output of split id to first input of split_id
        model.add_edge(einsum_id, split_id, vec![(0, 0)])?;
        model.connect_model_input(1, NodeInput::new(split_id, 1))?;
        model.connect_model_input(2, NodeInput::new(split_id, 2))?;

        model.add_consecutive_layer(Layer::Recombination(recombination_layer), Some(split_id))?;

        model.automatic_output_labelling()?;

        let inputs = model
            .input_shapes()
            .iter()
            .map(Tensor::random)
            .collect_vec();

        prove_quantized_model(model, inputs, &mut GenStore::default())?;
        Ok(())
    }
}
