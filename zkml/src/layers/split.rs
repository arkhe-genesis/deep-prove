use std::iter::once;

use anyhow::ensure;
use ark_ff::PrimeField;
use dp_crypto::{
    Expression, IntoMLE,
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::{ceil_log2, optimal_sumcheck_threads},
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::{Itertools, repeat_n};
use serde::{Deserialize, Serialize};
use tract_onnx::tract_hir::internal::num_integer::div_ceil;

use crate::{
    Claim, Element, InitTranscript, NextPowerOfTwo, Prover, Shape, VectorTranscript,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx},
    },
    model::Step,
    padding::{PaddingMode, ShapeInfo},
    poly_commit::identity_eval,
    tensor::{TensorTypeParam, WrappedTensor},
    try_unzip,
};

pub(crate) const SPLIT_LAYER: &str = "SPLIT";

/// A layer that splits each input tensor into the given number of chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitLayer {
    pub(crate) unpadded_input_shapes: Vec<Shape>, /* we need the unpadded input shapes to properly compute padded output shapes */
    pub(crate) num_chunks: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct SplitLayerSumcheckProof<F: PrimeField> {
    proof: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    evals: Vec<F>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct SplitLayerProof<F: PrimeField> {
    proofs: Vec<Option<SplitLayerSumcheckProof<F>>>,
}

impl SplitLayer {
    /// Find the number of chunks to split the input with the provided `unpadded_input_shape`,
    /// ensuring that at most `max_num_chunks` are employed, and that the chunk size is a
    /// power of two (to allow for efficient proving and verification)
    pub(crate) fn num_chunks_for_input_shape(
        max_num_chunks: usize,
        unpadded_input_shape: &Shape,
    ) -> anyhow::Result<usize> {
        let first_dim = unpadded_input_shape.dim(0);
        let mut num_chunks = max_num_chunks;
        let chunk_size =
            |num_chunks: usize| first_dim.next_power_of_two() / num_chunks.next_power_of_two();
        while chunk_size(num_chunks) * num_chunks < first_dim {
            // we'd have a too small chunk size for the current number of chunks, so we halve them
            num_chunks = num_chunks.next_power_of_two() / 2;
        }
        // now, we know that this chunk size is the smallest one that allows to cover the first dimension, so we
        // found the minimum number of chunks needed to cover it
        let chunk_size = chunk_size(num_chunks);
        num_chunks = div_ceil(first_dim, chunk_size);
        ensure!(
            num_chunks <= max_num_chunks,
            "Cannot instantiate a SplitLayer with {max_num_chunks} chunks for the provided input shape: {unpadded_input_shape:?}"
        );
        Ok(num_chunks)
    }

    /// Instantiate a new `SplitLayer` for the given `unpadded_input_shapes`, considering that the inputs
    /// should be split in at most `max_num_chunks`` chunks
    pub fn new_from_input_shapes(
        max_num_chunks: usize,
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<Self> {
        ensure!(
            max_num_chunks >= 2,
            "Doesn't make sense to chunk with less than 2 chunks"
        );
        let num_chunks = unpadded_input_shapes
            .iter()
            .map(|shape| Self::num_chunks_for_input_shape(max_num_chunks, shape))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            unpadded_input_shapes: unpadded_input_shapes.to_vec(),
            num_chunks,
        })
    }

    pub(crate) fn check_is_splittable(
        &self,
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<()> {
        // we assume we are always going to split the input tensors over the first dimension
        ensure!(unpadded_input_shapes.len() == self.num_chunks.len());
        for (shape, &num_chunks) in unpadded_input_shapes.iter().zip(&self.num_chunks) {
            let chunk_size = shape.dim(0).next_power_of_two() / num_chunks.next_power_of_two();
            ensure!(
                chunk_size * num_chunks >= shape.dim(0),
                "Too few number of chunks ({num_chunks}) for shape {shape:?}"
            );
        }
        Ok(())
    }

    fn unpadded_output_shapes(
        &self,
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<Vec<Shape>> {
        self.check_is_splittable(unpadded_input_shapes)?;
        Ok(unpadded_input_shapes
            .iter()
            .zip(&self.num_chunks)
            .flat_map(|(shape, &num_chunks)| {
                let chunk_size = shape.dim(0).next_power_of_two() / num_chunks.next_power_of_two();
                repeat_n(
                    Shape::from_it(
                        shape.iter().enumerate().map(
                            |(i, &dim)| {
                                if i == 0 { chunk_size } else { dim }
                            },
                        ),
                    ),
                    num_chunks - 1,
                )
                .chain(once(Shape::from_it(shape.iter().enumerate().map(
                    |(i, &dim)| {
                        if i == 0 {
                            if dim % chunk_size == 0 {
                                chunk_size
                            } else {
                                dim % chunk_size
                            }
                        } else {
                            dim
                        }
                    },
                ))))
            })
            .collect::<Vec<Shape>>())
    }

    fn padded_output_shapes(
        &self,
        input_shapes: &[Shape],
        unpadded_input_shapes: &[Shape],
    ) -> anyhow::Result<Vec<Shape>> {
        ensure!(
            input_shapes
                == unpadded_input_shapes
                    .iter()
                    .map(|shape| shape.next_power_of_two())
                    .collect_vec(),
            "Input shapes are not equal to pad(unpadded_input_shapes)"
        );
        let unpadded_output_shapes = self.unpadded_output_shapes(unpadded_input_shapes)?;
        Ok(unpadded_output_shapes
            .into_iter()
            .map(|shape| shape.next_power_of_two())
            .collect_vec())
    }

    fn recombine_claims<F: PrimeField, T: Transcript>(
        num_chunks: usize,
        point: &[F],
        chunk_evals: &[F],
        transcript: &mut T,
    ) -> Claim<F> {
        // append point and evals to transcript
        transcript.append_scalars(point);
        transcript.append_scalars(chunk_evals);
        // squeeze additional coordinates to recombine claims
        let num_additional_vars = ceil_log2(num_chunks);
        let challenges = transcript.read_challenges(num_additional_vars);
        let beta_evals = evals(&challenges);

        let combined_eval = chunk_evals
            .iter()
            .zip(beta_evals)
            .fold(F::ZERO, |acc, (&eval, beta_eval)| acc + eval * beta_eval);

        Claim::new(
            point.iter().copied().chain(challenges).collect(),
            combined_eval,
        )
    }

    fn sumcheck_expression<F: PrimeField>(num_chunks: usize) -> Expression<F> {
        (0..num_chunks).fold(Expression::Constant(F::ZERO), |expr, i| {
            expr + Expression::Challenge(0, i, F::ONE, F::ZERO)
                * Expression::WitIn(i as u16)
                * Expression::WitIn((i + num_chunks) as u16)
        })
    }

    fn split_output_claims<'a, 'b, F: PrimeField>(
        &self,
        last_claims: &'a [&'b Claim<F>],
    ) -> Vec<&'a [&'b Claim<F>]> {
        self.num_chunks
            .iter()
            .scan(0, |acc, &num_chunks| {
                let claims_slice = &last_claims[*acc..*acc + num_chunks];
                *acc += num_chunks;
                Some(claims_slice)
            })
            .collect()
    }
}

impl OpInfo for SplitLayer {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> anyhow::Result<Vec<Shape>> {
        match padding_mode {
            PaddingMode::NoPadding => self.unpadded_output_shapes(input_shapes),
            PaddingMode::Padding => {
                self.padded_output_shapes(input_shapes, &self.unpadded_input_shapes)
            }
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> anyhow::Result<usize> {
        ensure!(num_inputs == self.num_chunks.len());
        Ok(self.num_chunks.iter().sum())
    }

    fn describe(&self) -> String {
        format!("Split(num_chunks={:?})", self.num_chunks)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl<T: TensorTypeParam> Evaluate<T> for SplitLayer {
    fn evaluate(&self, inputs: &[&WrappedTensor<T>]) -> anyhow::Result<LayerOut<T>> {
        let outputs = inputs
            .iter()
            .zip(&self.num_chunks)
            .map(|(&input, &num_chunks)| {
                let new_shape = once(input.shape()[0].next_power_of_two())
                    .chain(input.shape()[1..].to_vec())
                    .collect_vec();
                let mut chunks = input
                    .clone()
                    .pad(new_shape.into(), T::zero())?
                    .chunk(num_chunks.next_power_of_two(), 0)?;
                chunks.truncate(num_chunks); // discard padding chunks
                // remove padding from last chunk
                let target_shape = once(chunks.last().unwrap().unpadded_shape()[0])
                    .chain(input.shape()[1..].to_vec())
                    .collect_vec()
                    .into();

                let unpadded_chunk = chunks.pop().unwrap().reduce_to_shape(&target_shape)?;
                chunks.push(unpadded_chunk);
                Ok(chunks)
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .concat();
        Ok(LayerOut::from_vec(outputs))
    }
}

impl ProveInfo for SplitLayer {
    fn step_info<F: PrimeField>(
        &self,
        mut aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        let output_shapes = self.output_shapes(&aux.last_output_shape, PaddingMode::Padding)?;
        aux.last_output_shape = output_shapes;
        let layer_ctx = LayerCtx::Split(self.clone());
        Ok((layer_ctx, aux))
    }
}

impl PadOp for SplitLayer {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self> {
        si.update_shapes(&self)?;
        Ok(self)
    }
}

impl<F, PCS> ProvableOp<F, PCS> for SplitLayer
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = SplitLayer;

    fn prove<'a, 'b, 'c, 'd, T: Transcript + InitTranscript>(
        &'a self,
        node_id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        // for each input, we need to recombine the corresponding output claims
        let (input_shapes, unpadded_input_shapes): (Vec<_>, Vec<_>) = step_data
            .inputs()
            .iter()
            .map(|handle| {
                (
                    handle.padded_shape().clone(),
                    handle.unpadded_shape().clone(),
                )
            })
            .unzip();
        self.check_is_splittable(&unpadded_input_shapes)?;
        let output_shapes = self.padded_output_shapes(&input_shapes, &unpadded_input_shapes)?;
        let num_outputs = output_shapes.len();
        ensure!(
            num_outputs == last_claims.len(),
            "Expected {num_outputs} output claims in SplitLayer, found {}",
            last_claims.len()
        );
        // split claims by corresponding input
        let claims_by_input = self.split_output_claims(&last_claims);

        // check if all claims for the same chunked input are on the same point or not; if not, we need a sumcheck to make them over
        // the same point
        let (input_claims, sumcheck_proofs): (Vec<_>, Vec<_>) = try_unzip(
            claims_by_input
                .into_iter()
                .zip(&self.num_chunks)
                .scan(0, |num_chunks_so_far, (claims, &num_chunks)| {
                    let output_index = *num_chunks_so_far;
                    *num_chunks_so_far += num_chunks;
                    Some((output_index, (claims, num_chunks)))
                })
                .map(|(output_index, (claims, num_chunks))| {
                    let point = claims[0].point();
                    let chunk_evals = once(claims[0].eval)
                        .chain(
                            claims[1..]
                                .iter()
                                .zip(&output_shapes[output_index + 1..output_index + num_chunks])
                                .filter_map(|(claim, out_shape)| {
                                    let chunk_num_vars = out_shape.num_vars().into_iter().sum();
                                    let (chunk_point, padding_point) =
                                        point.split_at(chunk_num_vars);
                                    if claim.point() == chunk_point {
                                        // we need to add `padding_point` coordinates to take into account
                                        // padding of the chunk to the size of the other padded chunks
                                        let eval = padding_point
                                            .iter()
                                            .fold(claim.eval, |eval, &padding_coord| {
                                                eval * (F::ONE - padding_coord)
                                            });
                                        Some(eval)
                                    } else {
                                        None
                                    }
                                }),
                        )
                        .collect_vec();
                    let output_shape = &output_shapes[output_index];
                    if chunk_evals.len() == claims.len() {
                        // all the claims are over the same point, so we simply recombine them
                        let input_claim = Self::recombine_claims(
                            num_chunks,
                            point,
                            &chunk_evals,
                            prover.transcript,
                        );
                        anyhow::Ok((input_claim, None))
                    } else {
                        // we need to do a sumcheck to make claims over the same point
                        let num_vars = output_shape.num_vars().into_iter().sum();
                        let num_threads = optimal_sumcheck_threads(num_vars);
                        let (output_mles, eq_polys): (Vec<_>, Vec<_>) =
                            try_unzip((0..num_chunks).map(|i| {
                                let mut out_tensor =
                                    step_data.output_tensor_at(output_index + i)?.clone();
                                out_tensor.pad_to_shape(output_shape.clone())?;
                                let output_mle = out_tensor.to_field_mle();
                                let padded_point = claims[i]
                                    .point()
                                    .iter()
                                    .cloned()
                                    .chain(std::iter::repeat(F::ZERO))
                                    .take(num_vars)
                                    .collect_vec();
                                let eq_poly = evals(&padded_point).into_mle();
                                // append claim to the transcript to further squeeze the challenge
                                prover.transcript.append_scalars(claims[i].point());
                                prover.transcript.append_scalars(&[claims[i].eval]);
                                anyhow::Ok((output_mle, eq_poly))
                            }))?;
                        let either_mles = output_mles
                            .iter()
                            .chain(&eq_polys)
                            .map(Either::Left)
                            .collect_vec();
                        let expr_builder = VirtualPolynomialsBuilder::<F>::new_with_mles(
                            num_threads,
                            num_vars,
                            either_mles,
                        );
                        let virtual_polys = expr_builder.to_virtual_polys(
                            &[Self::sumcheck_expression(num_chunks)],
                            &[prover.transcript.append_and_sample(b"claims_batching")],
                        );
                        let (proof, state) =
                            IOPProverState::prove(virtual_polys, prover.transcript);
                        let chunk_evals = &state.get_mle_flatten_final_evaluations()[..num_chunks];
                        let sumcheck_point = state.collect_raw_challenges();
                        let input_claim = Self::recombine_claims(
                            num_chunks,
                            &sumcheck_point,
                            chunk_evals,
                            prover.transcript,
                        );
                        Ok((
                            input_claim,
                            Some(SplitLayerSumcheckProof {
                                proof,
                                evals: chunk_evals.to_vec(),
                            }),
                        ))
                    }
                }),
        )?;

        prover.push_proof(
            node_id,
            LayerProof::SplitLayer(SplitLayerProof {
                proofs: sumcheck_proofs,
            }),
        );

        Ok(input_claims)
    }
}

impl<F, PCS> VerifiableCtx<F, PCS> for SplitLayer
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    type Proof = SplitLayerProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        _node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        // for each input, we need to recombine the corresponding output claims
        let input_shapes = shape_step.padded_input_shape.iter().cloned().collect_vec();
        let unpadded_input_shapes = shape_step
            .unpadded_input_shape
            .iter()
            .cloned()
            .collect_vec();
        let output_shapes = self.padded_output_shapes(&input_shapes, &unpadded_input_shapes)?;
        let num_outputs = output_shapes.len();
        ensure!(
            num_outputs == last_claims.len(),
            "Expected {num_outputs} output claims in SplitLayer, found {}",
            last_claims.len()
        );
        let claims_by_input = self.split_output_claims(last_claims);
        let input_claims = claims_by_input.into_iter().zip(&self.num_chunks)
        .scan(0, |num_chunks_so_far, (claims, &num_chunks)| {
            let output_index = *num_chunks_so_far;
            *num_chunks_so_far += num_chunks;
            Some((output_index,(claims, num_chunks)))
        })
        .enumerate()
        .map(|(input_index,(output_index, (claims, num_chunks)))| {
            let point = claims[0].point();
            let chunk_evals = once(claims[0].eval).chain(
                claims[1..].iter().zip(
                    &output_shapes[output_index + 1..output_index + num_chunks]
                    ).filter_map(|(claim, out_shape)| {
                        let chunk_num_vars = out_shape.num_vars().into_iter().sum();
                        let (chunk_point, padding_point) = point.split_at(chunk_num_vars);
                        if claim.point() == chunk_point {
                            // we need to add `padding_point` coordinates to take into account
                            //padding of the chunk to the size of the other padded chunks
                            let eval = padding_point
                                .iter()
                                .fold(claim.eval, |eval, &padding_coord| {
                                    eval * (F::ONE - padding_coord)
                                });
                            Some(eval)
                        } else {
                            None
                        }
                    })
                ).collect_vec();
                let output_shape = &output_shapes[output_index];
                if chunk_evals.len() == claims.len() {
                    // all the claims are over the same point, so we simply recombine them
                    ensure!(proof.proofs[input_index].is_none(), "Found sumcheck proof for input {input_index} of SplitLayer, even if the claims for chunked outputs are on the same point");
                    let input_claim = Self::recombine_claims(num_chunks, point, &chunk_evals, verifier.transcript);
                    anyhow::Ok(input_claim)
                } else {
                    ensure!(proof.proofs[input_index].is_some(), "Expected sumcheck proof for input {input_index} of SplitLayer");
                    let proof = proof.proofs[input_index].as_ref().unwrap();
                    let num_vars = output_shape.num_vars().into_iter().sum();
                    let (points, chunk_evals): (Vec<_>, Vec<_>) = claims.iter().map(|claim| {
                        let padded_point = claim.point().iter().cloned().chain(std::iter::repeat(F::ZERO)).take(num_vars).collect_vec();
                        verifier.transcript.append_scalars(&claim.point);
                        verifier.transcript.append_scalars(&[claim.eval]);
                        (padded_point, claim.eval)
                    }).unzip();
                    let challenge: F = verifier.transcript.append_and_sample(b"claims_batching");
                    let combined_eval = chunk_evals.into_iter().fold((F::ZERO, F::ONE), |(acc, chal), eval|
                        (acc + chal*eval, chal*challenge)
                    ).0;
                    let aux_info = VPAuxInfo {
                        max_degree: 2,
                        max_num_variables: num_vars,
                        ..Default::default()
                    };
                    let subclaim = IOPVerifierState::<F>::verify(
                        combined_eval,
                        &proof.proof,
                        &aux_info,
                        verifier.transcript,
                    );
                    let sumcheck_point = &subclaim.point;
                    // compute evals for eq polys
                    let calc_eval = points.into_iter().zip(&proof.evals)
                        .fold((F::ZERO, F::ONE), |(acc, chal), (point, &eval)| {
                            let eq_eval = identity_eval(sumcheck_point, &point);
                            (acc + chal*eval*eq_eval, chal*challenge)
                        }).0;
                    ensure!(
                        calc_eval == subclaim.expected_evaluation,
                        "SplitLayer verification failed, expected evaluation {:?} got {:?}",
                        subclaim.expected_evaluation,
                        calc_eval
                    );

                    let input_claim = Self::recombine_claims(num_chunks, sumcheck_point, &proof.evals, verifier.transcript);
                    Ok(input_claim)
                }
        }).collect::<anyhow::Result<Vec<_>>>()?;

        Ok(input_claims)
    }

    fn write_proof_to_transcript<T: Transcript>(
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
        Element, NextPowerOfTwo, Shape, Tensor,
        graph::NodeInput,
        layers::{Layer, einsum::EinSum, provable::OpInfo, split::SplitLayer},
        model::{Model, test::prove_quantized_model},
        padding::PaddingMode,
        rng_from_env_or_random,
        tensor::KeyedTensor,
    };

    #[test]
    fn test_split_layer() {
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

        let einsum_id = model
            .graph_mut()
            .add_inner(Layer::EinSum(
                EinSum::new(
                    "A(ijk)@B(kl)->O(ijl)".into(),
                    vec![Some(weight.into())],
                    vec![None],
                )
                .unwrap(),
            ))
            .unwrap();

        model
            .connect_model_input(0, NodeInput::new(einsum_id, 0))
            .unwrap();

        let unpadded_input_shapes = model.input_shapes();
        let max_num_chunks = 4usize;
        let split_id = model
            .graph_mut()
            .add_inner(Layer::Split(
                SplitLayer::new_from_input_shapes(max_num_chunks, &unpadded_input_shapes).unwrap(),
            ))
            .unwrap();

        // route output of split id to first input of split_id
        model.add_edge(einsum_id, split_id, vec![(0, 0)]).unwrap();
        model
            .connect_model_input(1, NodeInput::new(split_id, 1))
            .unwrap();
        model
            .connect_model_input(2, NodeInput::new(split_id, 2))
            .unwrap();

        model.automatic_output_labelling().unwrap();

        let inputs = model
            .input_shapes()
            .iter()
            .map(Tensor::random)
            .collect_vec();

        let unpadded_output_shapes = model
            .graph()
            .node(split_id)
            .unwrap()
            .as_inner()
            .unwrap()
            .output_shapes(&unpadded_input_shapes, PaddingMode::NoPadding)
            .unwrap();

        let padded_input_shapes = unpadded_input_shapes
            .into_iter()
            .map(|shape| shape.next_power_of_two())
            .collect_vec();
        let padded_output_shapes = model
            .graph()
            .node(split_id)
            .unwrap()
            .as_inner()
            .unwrap()
            .output_shapes(&padded_input_shapes, PaddingMode::Padding)
            .unwrap();

        let outputs = prove_quantized_model(model, inputs, &mut GenStore::default()).unwrap();

        for ((out, unpad_shape), out_shape) in outputs
            .into_iter()
            .zip(unpadded_output_shapes)
            .zip(padded_output_shapes)
        {
            assert_eq!(out.unpadded_shape().clone(), unpad_shape,);
            assert_eq!(out.shape().next_power_of_two(), out_shape,);
        }
    }
}
