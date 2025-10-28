//! Implementation of the RMSNorm layer
use crate::{
    Claim, Element, ProverContext, ScalingFactor, ScalingStrategy, Tensor,
    commit::{compute_betas_eval, identity_eval},
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        prover::Prover,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof, Requant,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            VerifiableCtx,
        },
    },
    lookup::{
        context::{
            COLUMN_SEPARATOR, LayerLookupContext, LookupWitnessGen, TableType, count_elements,
        },
        logup_gkr::{
            prover::batch_multiple_sizes_prove, structs::LogUpBatchProof,
            verifier::verify_logup_proof_multiple_sizes,
        },
    },
    model::StepData,
    number::Number,
    padding::PaddingMode,
    parser::{
        gguf::FileTensorLoader,
        json,
        llm::{LLMConfig, LLMVariant},
    },
    quantization::{self, Fieldizer},
    shape::Shape,
    tensor::{KeyedTensor, TensorKey, TensorTypeParam, WrappedTensor},
    to_base,
};
use anyhow::{Result, anyhow, ensure};
use ark_std::Zero;
use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression,
    mle::{IntoMLE, MultilinearExtension},
    util::{ceil_log2, transpose},
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
    virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use sumcheck::{
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use tracing::trace;
use transcript::Transcript;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

/// The short name used to identify the RMSNorm layer.
pub(crate) const RMSNORM_LAYER: &str = "RMSN";

/// The base 2 logarithm of the scale factor used in the inverse square root lookup tables
pub(crate) const LOG_RMSNORM_SCALE_FACTOR: usize = 20;
/// The scale factor for our fixed point arithmetic
pub(crate) const RMSNORM_SCALE_FACTOR: usize = 1 << LOG_RMSNORM_SCALE_FACTOR;
/// The scale factor of the outputs of the inverse square root lookup tables lookup
pub(crate) const RMSNORM_OUTPUT_SCALE_FACTOR: usize = 1 << 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Struct storing all information needed to perform RMSNorm. The `alpha` field
/// is normally learned parameters that are applied elementwise. The `eps` field is used for normalisation when calculating
/// the inverse square root.
pub struct RMSNorm<N> {
    /// Each element of the normalisation dimension is multiplied elementwise by this, we use
    /// an [`Option`] because it may be the case the weights are all 1 and then we don't want to apply this tensor.
    pub alpha: Option<KeyedTensor<N>>,
    /// Normalisation factor
    pub eps: f32,
    /// The size of the dimension we normalise over
    pub dim_size: usize,
    /// Contains information needed to perform quantised evaluation
    pub quant_info: Option<QuantisedRMSNormData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
/// This struct is used to store information used when evaluating the quantised version of [`RMSNorm`] on
/// [`Element`]s.
pub struct QuantisedRMSNormData {
    /// The [`ScalingFactor`] of the inputs
    input_scale_factor: ScalingFactor,
    /// This is the multiplier we have to rescale the inputs with
    multiplier: Element,
    /// This stores the [`RMSTableData`]
    lut: RMSTableData,
    /// The size of the dimension we average over
    dim_size: usize,
    /// This is the number of bits that get range checked
    range_check_bits: usize,
    /// The base 2 log of the value we have to multiply the most significant range check chunk by
    top_chunk_scalar_log: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
/// Struct used to store Softmax table data
pub struct RMSTableData {
    /// This is the result of calling [`f32::to_bits`] on the epsilon value.
    eps_bits: u32,
    /// The the number of bits to shift left by.
    pub(crate) range_check_bits: usize,
    /// The size of the dimension we normalise over
    pub(crate) dim_size: usize,
}

impl RMSTableData {
    pub(crate) fn new(eps_bits: u32, range_check_bits: usize, dim_size: usize) -> RMSTableData {
        RMSTableData {
            eps_bits,
            range_check_bits,
            dim_size,
        }
    }

    pub(crate) fn float_epsilon(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }

    pub(crate) fn table_output(&self, j: Element) -> Element {
        let epsilon = self.float_epsilon();
        // First we have to shift by `range_checked_bits`
        let shifted_val = j << self.range_check_bits;
        // Now we convert back to float and perform the operation
        let float_output = 1.0f32
            / ((shifted_val as f32 / (self.dim_size as f32 * RMSNORM_SCALE_FACTOR as f32))
                + epsilon)
                .sqrt();
        // Now we use the output scale factor to recover the element value
        (float_output * RMSNORM_OUTPUT_SCALE_FACTOR as f32).round() as Element
    }
}

impl<N: Number> RMSNorm<N> {
    /// Create a new [`RMSNorm`] layer with the given `alpha` and `eps` values.
    pub fn new(alpha: Option<KeyedTensor<N>>, eps: f32, dim_size: Option<usize>) -> Result<Self> {
        if alpha.is_none() && dim_size.is_none() {
            return Err(anyhow::anyhow!("Must provide either alpha or dim_size"));
        }
        // Unwrap is safe because we check one of alpha or dim_size is Some
        let dim_size = dim_size.unwrap_or_else(|| alpha.as_ref().map(|a| a.shape()[0]).unwrap());
        Ok(Self {
            alpha,
            eps,
            dim_size,
            quant_info: None,
        })
    }

    /// Returns the [`QuantisedRMSNormData`] if there is any.
    pub fn quant_info(&self) -> Option<&QuantisedRMSNormData> {
        self.quant_info.as_ref()
    }

    /// Returns the size of the dimension we normalise over.
    pub fn normalisation_dim_size(&self) -> usize {
        self.dim_size
    }

    /// Quantise the layer. To do this we want to have a common scale factor so that lookup tables can be reused, so we use the
    /// constant [`RMSNORM_SCALE_FACTOR`] as the input column scale factor. We need to work out how big the table needs to be to cover
    /// all of our possible inputs.
    ///
    /// This method returnss the quantised [`RMSNorm`] as well as the `intermediate_bit_size` for the following requant layer.
    pub fn quantise(
        &self,
        input_scaling: ScalingFactor,
        model_scaling: ScalingFactor,
    ) -> Result<(RMSNorm<Element>, usize)> {
        // The lookup input is SUM row_i * row_i, so we square every element and then sum them

        // Get the input scale
        let input_scale = input_scaling.scale();
        // Get the dim size (N)
        let dim_size = self.normalisation_dim_size();
        // We work out what we have to multiply by so that everything is scaled to `RMSNORM_SCALE_FACTOR` in quantised world
        let multiplier =
            (RMSNORM_SCALE_FACTOR as f32 * input_scale * input_scale).round() as Element;
        // Work out the number of variables the table requires, this is likely to be far too large to actually materialise as a table
        let full_table_bit_size = ceil_log2(dim_size)
            + 2 * (*quantization::BIT_LEN - 1)
            + ceil_log2(multiplier as usize)
            + 1;
        // To get around this we use the fact that we should only have roughly `2*(*quantization::BIT_LEN -1)` bits of precision i.e. only the most significant `2*(*quantization::BIT_LEN -1)`
        // can actually be "trusted" the rest are essentially junk because they don't come from the actual inputs and are just guesses at the part that we have already "rounded away" in quantisation.
        // So the actual part we perform inverse square root on is size `2*(*quantization::BIT_LEN -1)` and then we just need the discarded part to be range checked (which we do via a separate lookup).
        let range_checked_bits = full_table_bit_size - 2 * (*quantization::BIT_LEN - 1);

        // The final chunk might be values with fewer than *quantization::BIT_LEN bits so we work out what we need to scale the value up by in order to use our standard range check table.
        let remainder_bits = range_checked_bits % *quantization::BIT_LEN;
        let top_chunk_scalar_log = if !remainder_bits.is_zero() {
            *quantization::BIT_LEN - remainder_bits
        } else {
            0
        };
        // Calculate the lookup table
        let table_max: Element = 1 << (2 * (*quantization::BIT_LEN - 1));
        let table_min = -table_max;

        let table_data = RMSTableData::new(self.eps.to_bits(), range_checked_bits, dim_size);

        let max_lut_value = (table_min..table_max)
            .map(|v| table_data.table_output(v).abs())
            .max()
            .unwrap();
        // The value is positive so we just convert to usize
        let max_lut_value_bits = ceil_log2(max_lut_value as usize);

        // Make the QuantisedRMSNormData
        let quant_info = QuantisedRMSNormData {
            input_scale_factor: input_scaling,
            multiplier,
            lut: table_data,
            dim_size,
            range_check_bits: range_checked_bits,
            top_chunk_scalar_log,
        };

        let quant_alpha = self.alpha.as_ref().map(|alpha| {
            alpha.new_map_tensor(|alpha| {
                let new_data = alpha
                    .iter()
                    .map(|v| {
                        let vf32 = v.to_f32()?;
                        Ok(model_scaling.quantize(&vf32))
                    })
                    .collect::<Result<Vec<Element>, anyhow::Error>>()
                    .expect("Converting an f32 to f32 and quantising should never fail");
                Tensor::<Element>::new(alpha.shape().clone(), new_data)
            })
        });

        // To calculate the intermediate bit size we have that the output is `self.alpha * input  * lookup_output`
        // So lets work out the left hand bit size
        let intermediate_bit_size: usize =
            2 * (*quantization::BIT_LEN - 1) + 2 + max_lut_value_bits;

        Ok((
            RMSNorm::<Element> {
                alpha: quant_alpha,
                eps: self.eps,
                dim_size: self.dim_size,
                quant_info: Some(quant_info),
            },
            intermediate_bit_size,
        ))
    }
}

impl RMSNorm<f32> {
    pub fn from_json(l: &json::FileTensorLoader, _c: &LLMConfig) -> anyhow::Result<Self> {
        trace!("from_json: current path: {:?}", l.prefix);
        let alpha = l.get_tensor("norm.weight")?;
        let eps = l.metadata_to_f32("norm_epsilon")?;
        Self::new(Some(alpha), eps, None)
    }
    // Replaces from_var_builder and from_tensor_loader
    // The 'loader' passed here is expected to be pre-scoped by the caller
    // (e.g., loader.pp("attn_") or loader.pp("ffn_"))
    // HACK NOTE: "stack" is a temporary measure to counter the fact we dont have full GQA support yet
    // so we stack K and V and therefore we need to stack the norms as well
    pub fn from_loader(
        loader: &FileTensorLoader,
        c: &LLMConfig,
        stack: bool,
    ) -> anyhow::Result<Self> {
        let mut alpha = loader.get_tensor("norm.weight")?;
        if matches!(c.variant, LLMVariant::Gemma3) && stack {
            alpha = alpha.map_tensor(|alpha| {
                let (it, _) = alpha.slice_on_dim(0);
                let data = it
                    .flat_map(|t| std::iter::repeat_n(t, c.num_heads).flatten())
                    .cloned()
                    .collect::<Vec<_>>();
                let mut shape = alpha.shape().clone();
                let new_dim = shape.dim(-1) * c.num_heads;
                shape.set_dim(-1, new_dim);
                Tensor::new(shape, data)
            });
        }
        // we can have any checks on the shape alpha here since it depends of the context
        // a RMSNorm after  Q doesn't have the same shape as a RMSNorm after K or inside FeedForward etc
        let eps = loader
            .metadata::<f32>(c.variant.norm_epsilon_key())
            .unwrap_or_default();
        Self::new(Some(alpha), eps, None)
    }
}

impl<N: TensorTypeParam> OpInfo for RMSNorm<N> {
    // https://docs.rs/burn/0.17.0/burn/nn/struct.RmsNorm.html#impl-RmsNorm%3CB%3E
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        "RMSNorm".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Evaluate<f32> for RMSNorm<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<f32>],
    ) -> anyhow::Result<LayerOut<f32, E>> {
        assert!(inputs.len() == 1);
        let input = inputs[0];
        ensure!(
            input.rank() == 2,
            "RMSNorm input must have shape [seq_len, embedding_size]: found {:?}",
            input.shape(),
        );
        let embedding_size = input.shape().dims[1];
        let alpha = WrappedTensor::try_from(
            &self
                .alpha
                .clone()
                .map(|alpha| alpha.tensor())
                .unwrap_or(Tensor::<f32>::one(Shape::new(vec![embedding_size]))),
        )?;

        let output =
            WrappedTensor::rms_norm_forward(input.clone(), embedding_size, self.eps as f64, alpha)?;
        Ok(LayerOut::from_tensor(output))
    }
}

impl Evaluate<Element> for RMSNorm<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element, E>> {
        // First we check to see if there is any quant_info, if not error
        ensure!(
            self.quant_info.is_some(),
            "Cannot perform quantised RMSNorm evaluation if self.quant_info is None"
        );
        // Ensure we have a single input
        ensure!(
            inputs.len() == 1,
            "RMSNorm should have a single input, had: {}",
            inputs.len()
        );
        let input = inputs[0];

        let QuantisedRMSNormData {
            multiplier,
            lut,
            range_check_bits,
            ..
        } = self.quant_info.as_ref().unwrap();

        // So we need to take the input data and calculate `multiplier * SUM (xi * xi)`
        let final_dim = *input.shape().dims.last().ok_or(anyhow!(
            "Cannot evaluate RMSNorm, input didn't have a shape"
        ))?;

        let output_data = Tensor::try_from(input.clone())?
            .get_data()
            .chunks(final_dim)
            .flat_map(|chunk| {
                let sum_squares = chunk.iter().map(|x| *x * *x).sum::<Element>();
                let full_value = multiplier * sum_squares;
                let inv_sqrt = full_value >> range_check_bits;
                let denominator = lut.table_output(inv_sqrt);
                if let Some(alpha) = self.alpha.as_ref() {
                    chunk
                        .iter()
                        .zip(alpha.iter())
                        .map(|(&v, &alpha)| alpha * v * denominator)
                        .collect::<Vec<Element>>()
                } else {
                    chunk
                        .iter()
                        .map(|&v| v * denominator)
                        .collect::<Vec<Element>>()
                }
            })
            .collect::<Vec<Element>>();

        let output_tensor =
            WrappedTensor::try_from(&Tensor::<Element>::new(input.shape().into(), output_data))?;
        Ok(LayerOut::from_tensor(output_tensor))
    }
}

impl QuantizeOp for RMSNorm<f32> {
    type QuantizedOp = RMSNorm<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        // First check we have one input_scaling
        ensure!(
            input_scaling.len() == 1,
            "Could not quantise RMSNorm, too many input scaling factors {}, expected 1",
            input_scaling.len()
        );
        let input_scaling_factor = input_scaling[0];
        // Now we construct the `model_scaling` from `self.alpha`
        let model_scaling = if let Some(alpha) = self.alpha.as_ref() {
            ScalingFactor::from_tensor(alpha, None)
        } else {
            ScalingFactor::default()
        };

        let (quantised_rmsnorm, intermediate_bit_size) =
            self.quantise(input_scaling_factor, model_scaling)?;

        let mut output_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for RMSNorm layer different from 1"
        );
        let output_scaling = output_scalings.pop().unwrap();
        // Make the requant layer
        let requant = Requant::from_scaling_factors(
            input_scaling_factor,
            model_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        Ok(QuantizeOutput::new(quantised_rmsnorm, vec![output_scaling]).with_requant(requant))
    }
}

impl PadOp for RMSNorm<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        let RMSNorm {
            alpha,
            eps,
            dim_size,
            quant_info,
        } = self;

        let padded_alpha = alpha.map(|a| a.map_tensor(|a| a.pad_next_power_of_two()));

        Ok(RMSNorm::<Element> {
            alpha: padded_alpha,
            eps,
            dim_size,
            quant_info,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "E: ExtensionField + DeserializeOwned")]
pub struct RMSNormCtx<E: ExtensionField> {
    node_id: NodeId,
    /// The result of calling [`f32::to_bits`] on the epsilon used for normalisation purposes
    eps: u32,
    /// The number of bits that get range checked (so we can know how many instances there are in the range lookup)
    range_check_bits: usize,
    /// The size of the dimension we normalise over (unpadded)
    dim_size: usize,
    /// The multiplier used to scale up inputs to the lookup table.
    multiplier: Element,
    /// The base 2 logarithm of the multiplier for the most significant chunk we range check
    top_chunk_scalar_log: usize,
    /// The lookup info for the layer
    lookup_ctx: LayerLookupContext,
    /// The sumcheck expression for verifying the lookup input and layer output are correctly calculated
    sumcheck_expression: Vec<Expression<E>>,
    alpha_key: Option<TensorKey>,
}

impl<E: ExtensionField> OpInfo for RMSNormCtx<E> {
    fn output_shapes(&self, input_shapes: &[Shape], _padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes.to_vec()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        num_inputs
    }

    fn describe(&self) -> String {
        format!(
            "RMSNormCtx(dimension size: {}, epsilon: {})",
            self.dim_size, self.eps
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
/// Proof for correct execution of a quantised [`RMSNorm`] operation.
pub struct RMSNormProof<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// The LogUp proofs for RMSNorm, they are ordered `inv_sqrt_lookup`, `range_lookup`.
    pub(crate) logup_proof: LogUpBatchProof<E>,
    /// Witness commitments for this layer
    pub(crate) commitment: PCS::Commitment,
    /// The IO proof that links all claims to `last_claim` and the input
    pub(crate) io_proof: IOPProof<E>,
    /// The claimed evaluations of the commitments
    pub(crate) io_evaluations: Vec<E>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> RMSNormProof<E, PCS> {
    pub(crate) fn get_lookup_data(&self) -> (Vec<E>, Vec<E>) {
        self.logup_proof.fractional_outputs()
    }
    /// Writes the proof commitment to the transcript
    pub(crate) fn write_commitment<T: Transcript<E>>(&self, transcript: &mut T) -> Result<()> {
        PCS::write_commitment(&self.commitment, transcript).map_err(|e| anyhow!("{e:?}"))
    }
}

impl ProveInfo for RMSNorm<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        if let Some(quant_info) = self.quant_info() {
            let QuantisedRMSNormData {
                multiplier,
                dim_size,
                range_check_bits,
                top_chunk_scalar_log,
                lut,
                ..
            } = quant_info;

            // Add the tables that RMSNorm requires
            aux.tables.insert(TableType::Range);
            aux.tables.insert(TableType::RMSTable(*lut));

            let num_range_checks = (*range_check_bits - 1) / *quantization::BIT_LEN + 1;
            let tables = vec![TableType::Range, TableType::RMSTable(*lut)];
            let instances_per_table = vec![num_range_checks, 1];

            let lookup_ctx = LayerLookupContext::new(tables, instances_per_table);

            // Add the Alpha commitments
            if let Some(alpha) = self.alpha.as_ref() {
                aux.model_polys = {
                    let mut model_polys = HashMap::new();
                    model_polys.insert(self.alpha.as_ref().unwrap().key(), alpha.data().to_vec());
                    Some(model_polys)
                };
            }

            aux.max_poly_len = aux
                .last_output_shape
                .iter()
                .fold(aux.max_poly_len, |acc, shapes| {
                    acc.max(shapes.next_power_of_two().product())
                });

            let expr = build_sumcheck_expression::<E>(*multiplier, *dim_size, self.alpha.is_none());
            // The output shape is the same as the input shape so we don't need to update it
            // return the LayerCtx and the updated ContextAux
            Ok((
                LayerCtx::RMSNorm(RMSNormCtx {
                    node_id: id,
                    eps: self.eps.to_bits(),
                    range_check_bits: *range_check_bits,
                    dim_size: *dim_size,
                    multiplier: *multiplier,
                    top_chunk_scalar_log: *top_chunk_scalar_log,
                    lookup_ctx,
                    sumcheck_expression: vec![expr],
                    alpha_key: self.alpha.as_ref().map(|a| a.key()),
                }),
                aux,
            ))
        } else {
            Err(anyhow!(
                "RMSNorm operation has not been quantised so no proving info available"
            ))
        }
    }
}

/// Builds the sumcheck expressions used in `RMSNorm` proving/verifying.
/// The [`Expression`] returned links the inverse square root lookup input to the layer input and the `last_claim.eval` to the inverse square root output.
fn build_sumcheck_expression<E: ExtensionField>(
    multiplier: Element,
    dim_size: usize,
    trivial_alpha: bool,
) -> Expression<E> {
    if trivial_alpha {
        // The first sumcheck expression
        // Define constant values needed
        let multiplier_field: E = multiplier.to_field();
        let dim_vars = ceil_log2(dim_size);
        let two_mul = E::from_canonical_u64(1 << dim_vars);

        // This first part links the lookup input to the layer input
        let mean_square_expr = Expression::WitIn(0)
            * Expression::WitIn(0)
            * Expression::Constant(Either::Right(multiplier_field * two_mul));

        // The output should be alpha * input * lookup_output
        let output_expr = Expression::WitIn(1)
            * Expression::WitIn(0)
            * Expression::Challenge(0, 1, E::ONE, E::ZERO);

        // Finally we should link the lookup output claim to the polynomial we use in this sumcheck
        let lookup_linking_expr =
            Expression::WitIn(1) * Expression::Challenge(0, 2, E::ONE, E::ZERO);

        // Finally there should be two `eq_polys`
        let sum_eq_expr = Expression::WitIn(2);
        let last_claim_eq = Expression::WitIn(3);

        sum_eq_expr * (mean_square_expr + lookup_linking_expr) + last_claim_eq * output_expr
    } else {
        // The first sumcheck expression
        // Define constant values needed
        let multiplier_field: E = multiplier.to_field();
        let dim_vars = ceil_log2(dim_size);
        let two_mul = E::from_canonical_u64(1 << dim_vars);

        // This first part links the lookup input to the layer input
        let mean_square_expr = Expression::WitIn(0)
            * Expression::WitIn(0)
            * Expression::Constant(Either::Right(multiplier_field * two_mul));

        // The alpha expression will be input ID 2
        let alpha_expr = Expression::WitIn(2);

        // The output should be alpha * input * lookup_output
        let output_expr = alpha_expr
            * Expression::WitIn(1)
            * Expression::WitIn(0)
            * Expression::Challenge(0, 1, E::ONE, E::ZERO);

        // Finally we should link the lookup output claim to the polynomial we use in this sumcheck
        let lookup_linking_expr =
            Expression::WitIn(1) * Expression::Challenge(0, 2, E::ONE, E::ZERO);

        // Finally there should be two `eq_polys`
        let sum_eq_expr = Expression::WitIn(3);
        let last_claim_eq = Expression::WitIn(4);

        sum_eq_expr * (mean_square_expr + lookup_linking_expr) + last_claim_eq * output_expr
    }
}

impl<E, PCS> ProvableOp<E, PCS> for RMSNorm<Element>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = RMSNormCtx<E>;

    fn prove<T: transcript::Transcript<E>>(
        &self,
        node_id: NodeId,
        ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let input_tensors = step_data.input_tensors(store)?;
        // Check there is a single input
        ensure!(
            input_tensors.len() == 1,
            "RMSNorm step should only have one input, received {}",
            input_tensors.len()
        );
        let input_mle: MultilinearExtension<E> = input_tensors[0].get_data().to_vec().into_mle();

        let (claims, proof) = self.prove_step(node_id, last_claims, ctx, input_mle, prover)?;
        // Add the proof to the proof list
        prover.push_proof(node_id, LayerProof::<E, PCS>::RMSNorm(proof));

        Ok(claims)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        step_data: &StepData<Element, E>,
        store: &mut GenStore,
    ) -> Result<LookupWitnessGen<E, PCS>> {
        let output_tensors = step_data.output_tensors(store)?;
        ensure!(
            step_data.node_inputs.len() == 1,
            "Found more than 1 input in inference step of RMSNorm layer"
        );
        ensure!(
            output_tensors.len() == 1,
            "Found more than 1 output in inference step of RMSNorm layer"
        );
        let input_tensor = step_data.input_tensor_at(0, store)?;
        self.lookup_witness(id, ctx, input_tensor)
    }
}

type ProveOut<E, PCS> = (Vec<Claim<E>>, RMSNormProof<E, PCS>);
impl RMSNorm<Element> {
    pub(crate) fn prove_step<E, T, PCS>(
        &self,
        node_id: NodeId,
        last_claims: Vec<&Claim<E>>,
        ctx: &RMSNormCtx<E>,
        input_poly: MultilinearExtension<E>,
        prover: &mut Prover<E, T, PCS>,
    ) -> Result<ProveOut<E, PCS>>
    where
        E: ExtensionField,
        T: transcript::Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        // Check we have the correct number of claims
        ensure!(
            last_claims.len() == 1,
            "RMSNorm only produces one output claim but got: {}",
            last_claims.len()
        );
        let last_claim = last_claims[0];

        let layer_commitment = prover.lookup_witness(node_id)?;
        let logup_inputs = ctx
            .lookup_ctx
            .create_logup_inputs::<PCS, E>(layer_commitment, &prover.challenge_storage)?;
        let layer_polys = PCS::get_arc_mle_witness_from_commitment(layer_commitment);
        let commitment = PCS::get_pure_commitment(layer_commitment);

        // Run the lookup proof
        let logup_batch_proof = batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

        let batching_challenge = prover
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let num_vars = input_poly.num_vars();
        let diff = num_vars - logup_batch_proof.output_claims()[0].point.len();
        let logup_vars = logup_batch_proof.output_claims()[0].point.len();
        let num_threads = optimal_sumcheck_threads(num_vars);

        let full_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(logup_batch_proof.output_claims()[0].point.iter().copied())
            .collect::<Vec<E>>();

        let input_eq = compute_betas_eval(&full_point).into_mle();
        let last_claim_eq = compute_betas_eval(&last_claim.point).into_mle();

        let inv_sqrt_poly_evals = layer_polys
            .last()
            .ok_or(anyhow!("No Layer Polys for Layer Norm so cannot prove"))?
            .get_base_field_vec();
        let inv_sqrt_poly = MultilinearExtension::<E>::from_evaluations_vec(
            num_vars,
            inv_sqrt_poly_evals
                .iter()
                .flat_map(|&v| vec![v; 1 << diff])
                .collect::<Vec<E::BaseField>>(),
        );

        let alpha_poly: MultilinearExtension<E> = if let Some(alpha) = self.alpha.as_ref() {
            std::iter::repeat_n(
                alpha
                    .get_data()
                    .iter()
                    .map(<Element as Fieldizer<E>>::to_field)
                    .collect::<Vec<E>>(),
                1 << logup_vars,
            )
            .flatten()
            .collect::<Vec<E>>()
            .into_mle()
        } else {
            MultilinearExtension::<E>::default()
        };

        let either_mles = if self.alpha.is_some() {
            [
                &input_poly,
                &inv_sqrt_poly,
                &alpha_poly,
                &input_eq,
                &last_claim_eq,
            ]
            .into_iter()
            .map(Either::Left)
            .collect::<Vec<Either<_, _>>>()
        } else {
            [&input_poly, &inv_sqrt_poly, &input_eq, &last_claim_eq]
                .into_iter()
                .map(Either::Left)
                .collect::<Vec<Either<_, _>>>()
        };

        let expr_builder =
            VirtualPolynomialsBuilder::<E>::new_with_mles(num_threads, num_vars, either_mles);
        let virtual_poly =
            expr_builder.to_virtual_polys(&ctx.sumcheck_expression, &[batching_challenge]);
        let (io_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);
        let io_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let io_evaluations = if self.alpha.is_some() {
            state.get_mle_flatten_final_evaluations()[..3].to_vec()
        } else {
            state.get_mle_flatten_final_evaluations()[..2].to_vec()
        };

        let input_eval = io_evaluations[0];
        let input_point = state
            .challenges
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();
        let input_claim = Claim::<E>::new(input_point, input_eval);

        // Add the commitment claims to the commitment prover
        let first_commit_claims = (
            logup_batch_proof.output_claims()[0].point.to_vec(),
            logup_batch_proof
                .output_claims()
                .iter()
                .take(logup_batch_proof.output_claims().len() - 1)
                .map(|claim| claim.eval)
                .collect::<Vec<E>>(),
        );
        let second_commit_claim = (io_point[diff..].to_vec(), vec![io_evaluations[1]]);

        prover.add_witness_claim(node_id, vec![first_commit_claims, second_commit_claim]);

        if self.alpha.is_some() {
            let common_claims = {
                let point = io_point.iter().take(diff).copied().collect::<Vec<E>>();
                let mut claims = HashMap::new();
                claims.insert(
                    self.alpha.as_ref().unwrap().key(),
                    Claim::<E>::new(point.clone(), io_evaluations[2]),
                );
                claims
            };
            prover.add_common_claims(node_id, common_claims);
        }

        let proof = RMSNormProof::<E, PCS> {
            logup_proof: logup_batch_proof,
            commitment,
            io_proof,
            io_evaluations,
        };

        Ok((vec![input_claim], proof))
    }

    /// Internal method for generating the lookup witness for a [`RMSNorm`] step.
    fn lookup_witness<E, PCS>(
        &self,
        id: NodeId,
        ctx: &ProverContext<E, PCS>,
        input_tensor: Tensor<Element>,
    ) -> Result<LookupWitnessGen<E, PCS>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let mut wit_gen = LookupWitnessGen::<E, PCS>::default();

        // We need to work out how many chunks to split the shifted away part into to be range checked
        let QuantisedRMSNormData {
            range_check_bits,
            top_chunk_scalar_log,
            lut,
            multiplier,
            ..
        } = self.quant_info().ok_or(anyhow!(
            "Could not prove RMSNorm because it had no quantisation data"
        ))?;
        let (range_check, lookup_input): (Vec<Element>, Vec<Element>) = input_tensor
            .get_data_into()
            .chunks(self.normalisation_dim_size().next_power_of_two())
            .map(|chunk| {
                let sum_squares = chunk.iter().map(|x| *x * *x).sum::<Element>();
                let full_value = *multiplier * sum_squares;
                let inv_sqrt = full_value >> range_check_bits;
                (full_value - (inv_sqrt << range_check_bits), inv_sqrt)
            })
            .unzip();
        let number_range_checks = (range_check_bits - 1) / *quantization::BIT_LEN + 1;

        // Split `range_check` into its constituent parts
        let range_mask: Element = (1 << *quantization::BIT_LEN) - 1;
        let top_chunk_scalar: Element = 1 << top_chunk_scalar_log;
        let mut range_checks = (0..number_range_checks)
            .into_par_iter()
            .map(|j| {
                if j != number_range_checks - 1 {
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            tmp & range_mask
                        })
                        .collect::<Vec<Element>>()
                } else {
                    // In the final chunk after being shifted everything has to get multiplied by 1 << top_chunk_scalar_log
                    range_check
                        .iter()
                        .map(|&elem| {
                            let tmp = elem >> (j * *quantization::BIT_LEN);
                            (tmp & range_mask) * top_chunk_scalar
                        })
                        .collect::<Vec<Element>>()
                }
            })
            .collect::<Vec<Vec<Element>>>();
        let range_elements_count =
            range_checks
                .iter()
                .fold(HashMap::<Element, u64>::new(), |mut acc, range_check| {
                    range_check
                        .iter()
                        .for_each(|v| *acc.entry(*v).or_default() += 1);
                    acc
                });

        let (lookup_output, merged_inv_sqrt): (Vec<Element>, Vec<Element>) = lookup_input
            .iter()
            .map(|l_input| {
                let out = lut.table_output(*l_input);
                (out, *l_input + out * COLUMN_SEPARATOR)
            })
            .unzip();

        let inv_sqrt_element_count = count_elements(merged_inv_sqrt);

        // Make the commitments to the lookups
        let width = 1 + number_range_checks;
        range_checks.push(lookup_input);
        let transposed = transpose(range_checks);
        let first_values = to_base::<E, _>(transposed.concat());
        let rmm1 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(first_values, width),
            InstancePaddingStrategy::Default,
        );
        let rmm2 = RowMajorMatrix::<E::BaseField>::new_by_inner_matrix(
            ceno_p3::matrix::dense::DenseMatrix::new(to_base::<E, _>(lookup_output), 1),
            InstancePaddingStrategy::Default,
        );
        let layer_commitment = ctx.commitment_ctx.batch_commit(vec![rmm1, rmm2])?;

        // Add the merged columns to the lookups lists
        wit_gen.insert_element_count(TableType::Range, range_elements_count);

        wit_gen.insert_element_count(TableType::RMSTable(*lut), inv_sqrt_element_count);

        // Insert the LogUpWitnesses
        wit_gen.insert_logup_witness(id, layer_commitment);
        Ok(wit_gen)
    }
}

impl<E, PCS> VerifiableCtx<E, PCS> for RMSNormCtx<E>
where
    E: ExtensionField + Serialize + DeserializeOwned,
    E::BaseField: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Proof = RMSNormProof<E, PCS>;

    fn verify<T: transcript::Transcript<E>>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<E>],
        verifier: &mut Verifier<E, T, PCS>,
        _shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<E>>> {
        // First we check that we only have one claim in `last_claims`
        ensure!(
            last_claims.len() == 1,
            "RMSNorm only outputs 1 claim, received {} while verifying RMSNorm step",
            last_claims.len()
        );

        let last_claim = last_claims[0];

        let RMSNormProof {
            logup_proof,
            commitment,
            io_proof,
            io_evaluations,
        } = proof;

        // Verify the lookup proof
        let batch_claim = verify_logup_proof_multiple_sizes(logup_proof, verifier.transcript)?;
        self.lookup_ctx
            .verify_logup_batch_claim(&batch_claim, &verifier.challenge_storage)?;

        // Now we squeeze the batching challenge
        let alpha = verifier
            .transcript
            .sample_and_append_challenge(b"batching")
            .elements;

        let poly_evals = batch_claim.poly_evals();
        let range_evals = &poly_evals[..poly_evals.len() - 2];
        let inv_sqrt_input_eval = poly_evals[poly_evals.len() - 2];
        let inv_sqrt_output_eval = poly_evals[poly_evals.len() - 1];

        let pow_two_multiplier = E::from_canonical_u64(1 << *quantization::BIT_LEN);
        let (partial_eval, power_two) = range_evals.iter().take(range_evals.len() - 1).fold(
            (
                inv_sqrt_input_eval * E::from_canonical_u64(1 << self.range_check_bits),
                E::ONE,
            ),
            |(acc, pow), &eval| (acc + eval * pow, pow * pow_two_multiplier),
        );
        // The last range evaluation has to be rescaled
        let top_chunk_scalar_inv = E::from_canonical_u64(1 << self.top_chunk_scalar_log).inverse();

        let claimed_sum = partial_eval
            + *range_evals.last().unwrap() * top_chunk_scalar_inv * power_two
            + alpha * (alpha * inv_sqrt_output_eval + last_claim.eval);
        // If io_evaluations has length 3 then alpha is non-trivial
        let max_degree = if io_evaluations.len() == 3 { 4 } else { 3 };
        let aux_info = VPAuxInfo {
            max_num_variables: last_claim.point.len(),
            max_degree,
            ..Default::default()
        };
        let io_subclaim =
            IOPVerifierState::<E>::verify(claimed_sum, io_proof, &aux_info, verifier.transcript);
        let io_point = io_subclaim
            .point
            .iter()
            .map(|c| c.elements)
            .collect::<Vec<E>>();

        let diff = io_point.len() - batch_claim.point().len();
        let full_point = std::iter::repeat_n(E::TWO.inverse(), diff)
            .chain(batch_claim.point().iter().copied())
            .collect::<Vec<E>>();

        let input_eq = identity_eval(&full_point, &io_point);
        let last_claim_eq = identity_eval(&last_claim.point, &io_point);
        let witnesses = io_evaluations
            .iter()
            .copied()
            .chain([input_eq, last_claim_eq])
            .collect::<Vec<E>>();

        let calc_claim = eval_by_expr_with_instance(&[], &witnesses, &[], &[], &[alpha], &self.sumcheck_expression[0]).right().ok_or(anyhow!("RMSNorm verification failed, first sumcheck expression did not evaluate to an extension field element"))?;

        ensure!(
            calc_claim == io_subclaim.expected_evaluation,
            "RMSNorm verification failed, calculated claim {:?} did not equal the expected IO evaluation {:?}",
            calc_claim,
            io_subclaim.expected_evaluation
        );

        // Now we add the commitments to the verifier
        let first_commit = (
            batch_claim.point().to_vec(),
            poly_evals[..poly_evals.len() - 1].to_vec(),
        );
        let second_commit = (io_point[diff..].to_vec(), vec![io_evaluations[1]]);

        verifier.commit_verifier.add_witness_claim(
            self.node_id,
            commitment.clone(),
            vec![first_commit, second_commit],
        );

        if let Some(key) = &self.alpha_key {
            ensure!(
                io_evaluations.len() == 3,
                "Evaluation for alpha MLE not found in RMSNorm proof"
            );
            let common_claims = {
                let point = io_point.iter().take(diff).copied().collect::<Vec<E>>();
                let mut claims = HashMap::new();
                claims.insert(
                    key.clone(),
                    Claim::<E>::new(point.clone(), io_evaluations[2]),
                );
                claims
            };
            verifier.add_common_claims(self.node_id, common_claims);
        }

        Ok(vec![Claim::<E>::new(io_point.to_vec(), io_evaluations[0])])
    }

    fn write_proof_to_transcript<T: Transcript<E>>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

#[cfg(test)]
mod tests {
    use ff_ext::GoldilocksExt2;

    use crate::{
        init_test_logging_default,
        layers::Layer,
        model::{Model, test::prove_model},
        tensor::is_close_with_tolerance,
    };

    use super::*;

    impl<N: Number> RMSNorm<N> {
        pub fn random(size: usize, layer_name: Option<TensorKey>) -> Self {
            let layer_name = layer_name.unwrap_or("rmsnorm".to_string().into());
            let alpha = KeyedTensor::new(
                format!("alpha_{layer_name}"),
                Tensor::<N>::random(&vec![size].into()),
            );
            let eps = 1e-5;
            Self::new(Some(alpha), eps, Some(size)).unwrap()
        }

        pub fn random_trivial_weights(size: usize) -> Self {
            let eps = 1e-5;
            Self::new(None, eps, Some(size)).unwrap()
        }
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_rmsnorm() {
        let rmsnorm = RMSNorm::random(1024, None);
        let input = Tensor::<f32>::new(vec![1, 1024].into(), vec![0.0; 1024]);
        let output = rmsnorm.evaluate::<E>(&[&input.as_wrapped()]).unwrap();
        assert_eq!(
            output.outputs[0].shape().clone(),
            vec![1_usize, 1024].into()
        );
        assert_eq!(output.outputs[0].get_data(), vec![0.0; 1024]);
    }

    #[test]
    fn test_quantise_rmsnorm() {
        let rmsnorm = RMSNorm::random(100, None);
        // Make a random float input tensor and derive the input ScalingFactor
        let input_tensor = Tensor::<f32>::random(&vec![2, 100].into());
        let input_scaling = ScalingFactor::from_tensor(&input_tensor, None);
        let model_scaling = ScalingFactor::from_tensor(rmsnorm.alpha.as_ref().unwrap(), None);
        // Construct the quantised RMSNorm
        let (quant_rmsnorm, _) = rmsnorm.quantise(input_scaling, model_scaling).unwrap();
        // We quantise the float input to obtain `quant_tensor` and then we dequantise to obtain `dequant_input`
        // this lets us run quantised evaluation and floating point evaluation and compare the outputs.
        let quant_tensor = input_tensor.to_quantized(&input_scaling);
        let dequant_input = quant_tensor.dequantize(&input_scaling);

        let dequant_output = rmsnorm
            .evaluate::<E>(&[&dequant_input.as_wrapped()])
            .unwrap()
            .outputs[0]
            .clone();

        let quant_output = quant_rmsnorm
            .evaluate::<E>(&[&quant_tensor.as_wrapped()])
            .unwrap()
            .outputs[0]
            .clone();

        let output_scale = input_scaling.scale()
            * model_scaling.scale()
            * (1.0f32 / RMSNORM_OUTPUT_SCALE_FACTOR as f32);
        let output_scaling = ScalingFactor::from_scale(output_scale, None);
        let quant_output_dequant = quant_output.to_native().dequantize(&output_scaling);
        let a = quant_output_dequant.get_data();
        let b = dequant_output.get_data();
        assert!(
            is_close_with_tolerance(a, &b, 5e-2_f32, 1e-1_f32),
            "Wasn't close enough to floating point version"
        );
    }

    #[test]
    fn test_rmsnorm_proving() {
        init_test_logging_default();
        let rmsnorm = RMSNorm::random(100, None);

        let mut model =
            Model::new_from_input_shapes(vec![vec![15, 100].into()], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::RMSNorm(rmsnorm), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[test]
    fn test_rmsnorm_proving_trivial_weights() {
        init_test_logging_default();
        let rmsnorm = RMSNorm::random_trivial_weights(100);

        let mut model =
            Model::new_from_input_shapes(vec![vec![15, 100].into()], PaddingMode::NoPadding);

        let _ = model
            .add_consecutive_layer(Layer::RMSNorm(rmsnorm), None)
            .unwrap();

        model.automatic_output_labelling().unwrap();
        model.describe();
        prove_model(model, &mut GenStore::default()).unwrap();
    }
}
