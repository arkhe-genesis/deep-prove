use super::{
    LayerCtx,
    provable::{
        Evaluate, LayerOut, NodeId, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp,
        QuantizeOutput,
    },
};
use crate::{
    Claim, Element, Prover, ScalingStrategy, Shape, VectorTranscript,
    backend::{Conv2dConfig, zkml_conv2d_i},
    commit::{compute_betas_eval, identity_eval},
    iop::{context::ContextAux, prover::BatchFFTProof},
    layers::{LayerProof, hadamard, provable::ProvingData, requant::Requant},
    model::StepData,
    padding::{PaddingMode, ShapeInfo},
    parser::{check_filter, safe_conv2d_shape},
    quantization::{self, BIT_LEN, Fieldizer, ScalingFactor, TensorFielder},
    shape::filter_size,
    tensor::{ConvData, ConvFFTData, Number, Tensor, fft},
    util::from_mle_list_dimensions,
};
use anyhow::{Context, Result, ensure};
use burn::tensor::{module::conv2d, ops::ConvOptions};
use core::f32;
use either::Either;
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{
    Expression, mle::IntoMLE, util::ceil_log2, virtual_polys::VirtualPolynomialsBuilder,
};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    prelude::*,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::HashMap, mem};
use sumcheck::{
    structs::{IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
};
use tenstore::GenStore;
use tracing::{info, warn};
use transcript::Transcript;

/// The short name used to identify the convolution layer
pub const CONVOLUTION_LAYER: &str = "CONV";

const IS_PROVABLE: bool = true;

pub(crate) mod proof;
#[cfg(test)]
mod test;

pub(crate) use proof::{ConvCtx, ConvProof};

const FILTER_POLY_ID: &str = "ConvFilter";
const BIAS_POLY_ID: &str = "ConvBias";

/// The filter weights, a 4D tensor of the shape `(feature_maps, channels_out,
/// kernel_height, kernel_width)` whose shape depends on the current life stage
/// of the filter.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum FilterTensor<T> {
    /// The stage-2, pow2-padded, filter tensor.
    RawFilter(Tensor<T>),
    /// The FFT-ized tensor, built from the above and the padded input shape.
    FftFilter {
        tensor: Tensor<T>,

        /// The originally padded shape of the filter, before it will have been
        /// converted in the shape adapted to the given inputs.
        pre_fft_shape: Shape,
    },
}
impl<T: Clone + Copy + Default> FilterTensor<T> {
    /// Ensure width and height are the same power-of-two.
    ///
    /// This prepares the filter to be used to compute a FFT. Note the shape may
    /// be altered without the underlying data being padded, the padding happens
    /// during layer evaluation.
    fn prepare_for_fft(&mut self, padded_input_shape: &Shape) {
        let FilterTensor::RawFilter(ref mut tensor) = self else {
            unreachable!("filter already ready for FFT")
        };

        // The ephemeral stage-2 filter tensor.
        tensor.pad_to_shape(tensor.shape().next_power_of_two());

        let pre_fft_shape = tensor.shape().clone();
        assert!(
            pre_fft_shape.product() == tensor.data().len(),
            "Shape does not match data length."
        );
        assert!(
            pre_fft_shape.rank() == 4,
            "Tensor shape does not match a convolution. expected 4 got {}",
            pre_fft_shape.rank(),
        );
        assert!(
            pre_fft_shape[2].is_power_of_two(),
            "Filter dimension is not power of two"
        );

        // n_w is the padded version of the input
        //
        // NOTE: The shape is modified here, but the data is not reallocated. The actual padding
        // is performed later on convolution evaluation, by the function [index_w];
        let n_w = (padded_input_shape[1] - pre_fft_shape[2] + 1).next_power_of_two();
        let new_shape = Shape::new(vec![pre_fft_shape[0], pre_fft_shape[1], n_w, n_w]);

        let tensor = Tensor::new_unchecked(new_shape, tensor.data().to_vec());

        *self = FilterTensor::FftFilter {
            tensor,
            pre_fft_shape,
        };
    }
}

/// The `Filter` encapsulates the operations related to the filter part of the
/// (filter, bias) that a convolution is.
///
/// The filter knows three stages of existence.
///
/// 1. The original, non-padded shape, directly as read from the model file.
///    This is kept track of in the [`FilterTensor::RawTensor`] variant of the
///    filter tensor.
///
/// 2. The intermediary shape, i.e. the pow2-padded version of the above. The
///    echo of its data lies in the stage-3 (cf. note below), and only its shape
///    remains, encoded in the [`pre_fft_shape`] of the stage-3 variant.
///
/// 3. The FFT-ready filter, built in conjunction from the above stage-2 filter
///    and the shape of the inputs, that is padded to the FFT requirements. NOTE:
///    this stage-3 filter has a mismatch between its data cardinality (kept intact
///    from stage-2), and its shape, which is *at least* as large as its stage-2
///    one, but typically larger.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Filter<T> {
    /// The stage-specific data of the tensor, either stage-2 or stage-3.
    tensor: FilterTensor<T>,

    /// The vestigial shape of the filter tensor, as originally defined in the
    /// model.
    original_shape: Shape,
}
impl<T> Filter<T> {
    /// Create a new filter, in its raw form, from the given [`Tensor`].
    fn new(tensor: Tensor<T>) -> Self {
        Self {
            original_shape: tensor.shape().clone(),
            tensor: FilterTensor::RawFilter(tensor),
        }
    }

    /// Return a view over the wrapped tensor for FFT computation. Panics if the
    /// wrapped tensor is not ready for FFT computation.
    fn as_pre_fft_tensor(&self) -> &Tensor<T> {
        match &self.tensor {
            FilterTensor::RawFilter(ref tensor) => tensor,
            FilterTensor::FftFilter { .. } => unreachable!("filter tensor is not in pre-FFT shape"),
        }
    }

    /// Return a view over the wrapped tensor for FFT computation. Panics if the
    /// wrapped tensor is not ready for FFT computation.
    fn as_fft_tensor(&self) -> (&Tensor<T>, &Shape) {
        match &self.tensor {
            FilterTensor::RawFilter(_) => unreachable!("filter tensor is not in FFT shape"),
            FilterTensor::FftFilter {
                ref tensor,
                ref pre_fft_shape,
            } => (tensor, pre_fft_shape),
        }
    }

    /// Return the stage-2 (i.e. pre-FFT-ization) shape of the filter tensor.
    fn pre_fft_shape(&self) -> &Shape {
        match &self.tensor {
            FilterTensor::RawFilter(tensor) => tensor.shape(),
            FilterTensor::FftFilter { pre_fft_shape, .. } => pre_fft_shape,
        }
    }
}
impl<T: Default + Clone + Copy> Filter<T> {
    /// Prepare the filter to be used in FFT computation by converting the
    /// encapsulated filter tensor from stage-1 to stage-3.
    fn prepare_for_fft(&mut self, padded_input_shape: &Shape) {
        self.tensor.prepare_for_fft(padded_input_shape);
    }
}
impl Filter<Element> {
    /// Convolution algorithm using FFTs. When invoking this algorithm the
    /// prover generates all witness/intermediate evaluations needed to generate
    /// a convolution proof of `self` applied over `input` with `bias`.
    fn fft_conv<F: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        bias: &Tensor<Element>,
    ) -> (Tensor<Element>, ConvData<F>) {
        /// Properly pad a filter
        ///
        /// We use this function so that filter is amenable to FFT based conv2d
        /// Usually vec and n are powers of 2
        ///
        /// Output: [[F[0][0],…,F[0][n_w],0,…,0],[F[1][0],…,F[1][n_w],0,…,0],…]
        fn index_w<E: ExtensionField>(
            w: &[Element],
            n_real: usize,
            n: usize,
            output_len: usize,
        ) -> impl ParallelIterator<Item = E> + use<'_, E> {
            (0..output_len).into_par_iter().map(move |idx| {
                let i = idx / n;
                let j = idx % n;
                if i < n_real && j < n_real {
                    w[i * n_real + j].to_field()
                } else {
                    E::ZERO
                }
            })
        }

        let (tensor, pre_fft_shape) = self.as_fft_tensor();

        // Sanity check, this layer must have been padded to perform the FFT
        assert_eq!(
            pre_fft_shape[0],
            tensor.dim(0),
            "The number of features maps must match after padding. original {:?} padded {:?}",
            pre_fft_shape,
            tensor.shape(),
        );
        assert_eq!(
            pre_fft_shape[1],
            tensor.dim(1),
            "The number of channels out must match after padding. original {:?} padded {:?}",
            pre_fft_shape,
            tensor.shape(),
        );
        assert!(
            pre_fft_shape[2] <= tensor.dim(2),
            "The padded width must be greater-than-equal the original width. original {:?} padded {:?}",
            pre_fft_shape,
            tensor.shape(),
        );
        assert!(
            pre_fft_shape[3] <= tensor.dim(3),
            "The padded height must be greater-than-equal the original height. original {:?} padded {:?}",
            pre_fft_shape,
            tensor.shape(),
        );
        assert!(
            tensor.dim(2).is_power_of_two(),
            "The padded width must be a power of two",
        );
        assert!(
            tensor.dim(3).is_power_of_two(),
            "The padded height must be a power of two",
        );
        assert_eq!(
            input.shape().rank(),
            3,
            "Only 3D input tensors are supported",
        );
        assert_eq!(
            input.dim(0),
            tensor.dim(1),
            "Grouping is not support, input and output channels must match. input {:?} padded {:?}",
            input,
            tensor.shape(),
        );
        assert_eq!(
            input.dim(1),
            input.dim(2),
            "Input must be square. shape {:?}",
            input.shape(),
        );

        let n_x = input.shape()[1].next_power_of_two();
        let real_input = input.to_field::<F>();
        let new_n = 2 * n_x * n_x;

        // Convert the convolution input to the frequency domain.
        //
        // This will also collect proving/debugging data.
        let (input_fft, input): (Vec<Vec<F>>, Vec<Vec<F>>) = real_input
            .par_chunks(n_x * n_x)
            .map(|chunk| {
                let xx_input = chunk.iter().cloned().rev().collect::<Vec<_>>();
                let mut xx_fft = xx_input
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(F::ZERO))
                    .take(new_n)
                    .collect::<Vec<_>>();
                fft(&mut xx_fft, false);
                (xx_fft, xx_input)
            })
            .unzip();

        let mut output = vec![vec![F::ZERO; 2 * filter_size(tensor.shape())]; tensor.dim(0)];

        // Convert the filter to frequency domain and perform the point-wise multiplication
        //
        // Compute a channel at the time.
        for (batch, batch_output) in output.iter_mut().enumerate().take(pre_fft_shape[0]) {
            for (channel, channel_input_fft) in input_fft.iter().enumerate().take(pre_fft_shape[1])
            {
                // The data range for a single channel
                let og_strides = pre_fft_shape.strides();
                let start = batch * og_strides[0] + channel * og_strides[1];
                let end = start + og_strides[1];

                let mut w_fft_temp = index_w(
                    &tensor.data()[start..end],
                    pre_fft_shape[2],
                    tensor.dim(2),
                    2 * filter_size(tensor.shape()),
                )
                .collect::<Vec<F>>();

                // Convert the convolution filter to frequency domain
                fft(&mut w_fft_temp, false);

                // Perform the point wise multiplication
                for k in 0..batch_output.len() {
                    batch_output[k] += channel_input_fft[k] * w_fft_temp[k];
                }
            }
        }

        // Convert the result back from the frequency domain
        let prod = output.clone();
        for elt in output.iter_mut() {
            fft(elt, true);
        }

        let mut conv_data = ConvData::new(real_input, input, input_fft, prod, output, n_x);
        let mut result = Tensor::new(
            vec![tensor.shape()[0], n_x, n_x].into(),
            conv_data.output_as_element.clone(),
        );
        assert_eq!(
            result.data().len(),
            result.shape().product(),
            "Result should have the correct number of elements",
        );

        for i in 0..result.dim(0) {
            for j in 0..filter_size(result.shape()) {
                let idx = i * filter_size(result.shape()) + j;
                result[idx] += bias[i];
            }
        }

        // Record here the output _after_ the bias addition. It's needed for
        // proving since we're proving the clearing garbage and that produces a
        // new claim on this output.
        //
        // XXX: Deentagle ConvData and the result.
        conv_data.set_output(result.get_data());

        (result, conv_data)
    }
}

/// Convolution layer description (weights)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Convolution<T> {
    /// The convolution kernel tensor. May be in raw form or ready for FFT.
    filter: Filter<T>,

    /// The convolution bias.
    ///
    /// This must have the same size as `feature_maps`.
    bias: Tensor<T>,
}
impl<T> Convolution<T> {
    pub fn new(filter: Tensor<T>, bias: Tensor<T>) -> Self {
        assert_eq!(bias.rank(), 1);
        assert_eq!(filter.dim(0), bias.shape()[0]);
        assert_eq!(filter.rank(), 4);
        Self {
            filter: Filter::new(filter),
            bias,
        }
    }

    pub(crate) fn output_shape(&self, input_shape: &Shape, padding_mode: PaddingMode) -> Shape {
        match padding_mode {
            // unpadded shape is the shape found in onxx file for example
            PaddingMode::NoPadding => conv2d_shape(input_shape, &self.filter.original_shape),
            PaddingMode::Padding => padded_conv2d_shape(input_shape, self.filter.pre_fft_shape()),
        }
    }

    /// Returns a reference to the bias data.
    fn bias(&self) -> &Tensor<T> {
        &self.bias
    }

    fn kw(&self) -> usize {
        self.filter.as_fft_tensor().0.dim(0)
    }

    fn kx(&self) -> usize {
        self.filter.as_fft_tensor().0.dim(1)
    }

    fn fft_filter_size(&self) -> usize {
        filter_size(self.filter.as_fft_tensor().0.shape())
    }

    fn pre_fft_filter_size(&self) -> usize {
        filter_size(self.filter.pre_fft_shape())
    }

    fn num_outputs(num_inputs: usize) -> usize {
        assert_eq!(num_inputs, 1);
        1
    }

    /// Returns this layers [ConvCtx].
    pub(crate) fn conv_context(&self, node_id: NodeId) -> ConvCtx {
        ConvCtx {
            node_id,
            kw: self.kw(),
            kx: self.kx(),
            nw: self.filter.as_fft_tensor().0.dim(2),
            real_nw: self.filter.pre_fft_shape()[2],
            filter_size: self.fft_filter_size(),
            unpadded_filter_shape: self.filter.original_shape.clone(),
            padded_filter_shape: self.filter.pre_fft_shape().clone(),
        }
    }
}
impl<T: Number> Convolution<T> {
    pub(crate) fn new_without_bias(filter: Tensor<T>) -> Self {
        let bias = Tensor::zeros(Shape::new(vec![filter.dim(0)]));
        Self::new(filter, bias)
    }
}
impl<T: Number> OpInfo for Convolution<T> {
    fn output_shapes(&self, input_shapes: &[Shape], padding_mode: PaddingMode) -> Vec<Shape> {
        input_shapes
            .iter()
            .map(|shape| self.output_shape(shape, padding_mode))
            .collect()
    }

    fn num_outputs(&self, num_inputs: usize) -> usize {
        Self::num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        format!("Conv: {:?}", self.filter.original_shape)
    }

    fn is_provable(&self) -> bool {
        IS_PROVABLE
    }
}
impl Evaluate<f32> for Convolution<f32> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<f32>],
        _unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<f32, E>> {
        let tensor = self.filter.as_pre_fft_tensor();
        ensure!(
            inputs.len() == 1,
            "Expected exactly 1 input when evaluating convolution layer, found {}",
            inputs.len(),
        );
        let input = inputs[0];
        ensure!(
            input.rank() == 3 || input.rank() == 4,
            "Input must be rank 3 or 4, got {}",
            input.rank(),
        );

        let input = if input.rank() == 3 {
            // Single batch
            input.clone().unsqueeze(0).to_btensor::<4>()
        } else {
            input.clone().to_btensor::<4>()
        };

        let weight = tensor.clone().to_btensor::<4>();
        let bias = self.bias.clone().to_btensor::<1>();

        let res = conv2d(
            input,
            weight,
            Some(bias),
            ConvOptions {
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
                groups: 1,
            },
        );

        let data = res
            .to_data()
            .into_vec()
            .expect("Failed to compute Convolution");

        Ok(LayerOut::from_vec(vec![Tensor::new(
            res.shape().into(),
            data,
        )]))
    }
}
impl Convolution<f32> {
    /// Quantizes the filter and the bias.
    /// It uses a custom scaling factor `bias_s` for the bias, if provided,
    /// otherwise the same scaling factor of the weights (i.e., `s`) is used
    fn quantize(self, s: &ScalingFactor, bias_s: &ScalingFactor) -> Convolution<Element> {
        let tensor = self.filter.as_pre_fft_tensor();
        let quantized_filter = tensor.to_quantized(s);
        let bias = self.bias.to_quantized(bias_s);
        Convolution::<Element>::new(quantized_filter, bias)
    }

    fn max_abs_weight(&self) -> f32 {
        let tensor = self.filter.as_pre_fft_tensor();
        let max_weight = tensor.max_abs_output();
        let max_bias = self.bias.max_abs_output();
        let distance = (max_weight - max_bias).abs() / max_weight;
        if distance > 0.1 {
            warn!(
                "max_abs_weight CONV: distance between max_weight and max_bias is too large: {:.2}%",
                distance * 100.0
            );
        }
        tensor.max_abs_output().max(self.bias.max_abs_output())
    }
}
impl Evaluate<Element> for Convolution<Element> {
    fn evaluate<E: ExtensionField>(
        &self,
        inputs: &[&Tensor<Element>],
        unpadded_input_shapes: &[Shape],
    ) -> Result<LayerOut<Element, E>> {
        ensure!(
            unpadded_input_shapes.len() == 1,
            "Expected exactly 1 input shape when evaluating convolution layer, got {}",
            unpadded_input_shapes.len(),
        );
        let unpadded_input_shape = &unpadded_input_shapes[0];
        ensure!(
            inputs.len() == 1,
            "Expected exactly 1 input when evaluating convolution layer, got {}",
            inputs.len(),
        );
        let input = inputs[0];
        ensure!(
            input.rank() == 3 || input.rank() == 4,
            "Input must be rank 3 or 4, got {}",
            input.rank(),
        );

        let (tensor, _) = self.filter.as_fft_tensor();

        // The filter and bias have been padded and converted to fft. Re-create
        // the tensors with original shapes.
        let mut filter = tensor.clone();

        // XXX: workaround for `into_fft_conv` not allocating underlying data,
        // without this change `copy_to_shape` perform index out-of-bounds.
        let _ = mem::replace(
            filter.shape_mut(),
            self.filter.original_shape.next_power_of_two(),
        );

        let kernels = filter.reduce_to_shape(&self.filter.original_shape);
        let bias = self
            .bias
            .reduce_to_shape(&Shape::new(vec![self.filter.original_shape[0]]));

        let input = input.reduce_to_shape(unpadded_input_shape);
        let input = if input.rank() == 4 {
            input.squeeze(0)
        } else {
            input
        };

        // The output is expected to be padded to the fft shape
        let n_x = input.dim(1).next_power_of_two();
        let fft_shape = Shape::new(vec![tensor.dim(0), n_x, n_x]);

        let kernels = kernels.to_btensor::<4>();
        let bias = bias.to_btensor::<1>();
        let input = input.to_btensor::<3>();
        let input = input.unsqueeze_dim(0);

        // Compute the convolution using the traditional convolution hardware accelerated.
        let config = Conv2dConfig { stride: 1 };
        let res = zkml_conv2d_i(input, kernels, bias, config);

        let conv_output = res
            .to_data()
            .into_vec()
            .expect("Failed to compute Convolution");

        let shape_out = Shape::from(res.shape());
        let conv_output = Tensor::new(shape_out, conv_output);

        let mut conv_output = conv_output.squeeze(0); // conv2d always return a 4D tensor
        conv_output.pad_to_shape(fft_shape);

        Ok(
            LayerOut::from_vec(vec![conv_output]).with_proving_data(ProvingData::Convolution(
                ConvFFTData {
                    input: inputs[0].clone(),
                    unpadded_input_shape: unpadded_input_shape.clone(),
                },
            )),
        )
    }
}
impl Convolution<Element> {
    /// Ensures filter and bias are of the correct shape to be used with [fft].
    ///
    /// The data is padded to a power of two because [fft] is a radix-2 implementation.
    ///
    /// NOTE: This must be called when padding the layer, this is because layers
    /// are chained together, with the output of one becoming the input of another
    /// and the other layers do expect the data to be padded. Unfortunately the
    /// padding has to be undone when computing the convolution during layer evaluation.
    ///
    /// NOTE: The filter's shape may be modified to ensure the width and height
    /// are the same, i.e. the filter is a square. In cases this does happen the
    /// filter data is not padded, this is performed during fft computation at layer
    /// evaluation.
    pub(crate) fn prepare_for_fft(&mut self, unpadded_input_shape: &Shape) {
        self.bias
            .pad_to_shape(self.bias.shape().next_power_of_two());

        self.filter
            .prepare_for_fft(&unpadded_input_shape.next_power_of_two());
    }

    /// Chainable version of [`prepare_for_fft`]
    pub fn prepared_for_fft(mut self, unpadded_input_shape: &Shape) -> Self {
        self.prepare_for_fft(unpadded_input_shape);
        self
    }

    /// Compute the convolution using FFT.
    ///
    /// See: https://en.wikipedia.org/wiki/Convolution_theorem
    fn fft<E: ExtensionField>(
        &self,
        input: &Tensor<Element>,
        unpadded_input_shape: &Shape,
    ) -> (Tensor<Element>, ConvData<E>) {
        let (conv_output, proving_data) = self.filter.fft_conv(input, &self.bias);

        let unpadded_output_shape = conv2d_shape(unpadded_input_shape, &self.filter.original_shape);
        debug_assert_eq!(
            padded_conv2d_shape(input.shape(), self.filter.pre_fft_shape()),
            *conv_output.shape(),
            "FFT output shape not computable"
        );

        // Set additional data due to padding to `0`.
        let cleared_tensor = clear_garbage(&conv_output, &unpadded_output_shape);

        (cleared_tensor, proving_data)
    }

    /// Returns the maximum bitsize of the output of this layer
    fn output_bitsize(&self) -> usize {
        // 2^{BIT_LEN + log2(k_h * k_w * k_c)}
        let (_k_n, k_c, k_h, k_w) = self.filter.as_pre_fft_tensor().get4d();
        2 * (*quantization::BIT_LEN - 1) + ceil_log2(k_h * k_w * k_c + 1)
    }

    fn prove_batch_fft_weights<
        E,
        T: Transcript<E>,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    >(
        &self,
        prover: &mut Prover<E, T, PCS>,
        r: Vec<E>,
    ) -> BatchFFTWeightsProof<E>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let (tensor, pre_fft_shape) = self.filter.as_fft_tensor();

        let padded_rows = 2 * self.fft_filter_size();
        let mut w1_reduced: Vec<E> = vec![E::ZERO; self.pre_fft_filter_size()];

        // Partition r in (r1,r2)
        let mut r1 = vec![E::ZERO; padded_rows.ilog2() as usize];
        let mut r2 = vec![E::ZERO; r.len() - padded_rows.ilog2() as usize];
        let r1_len = r1.len();
        r1.copy_from_slice(&r[..r1_len]);

        for i in 0..r2.len() {
            r2[i] = r[i + r1.len()];
        }

        // compute W(r1,i)
        let mut w_red: Vec<E> = vec![E::ZERO; padded_rows];
        let mut f_middle: Vec<Vec<E>> = vec![Vec::new(); r1.len() - 1];
        let beta = compute_betas_eval(&r2);
        Prover::<E, T, PCS>::phi_g_init(
            &mut w_red,
            &mut f_middle,
            r1.clone(),
            E::ONE,
            padded_rows.ilog2() as usize,
            false,
        );

        // compute X(i,r2)
        let filter_size = filter_size(pre_fft_shape);
        (0..tensor.dim(0)).for_each(|i| {
            (0..tensor.dim(1)).for_each(|j| {
                (0..filter_size).for_each(|k| {
                    let index = i * filter_size * tensor.dim(1) + j * filter_size + k;
                    let v: E = tensor[index].to_field();
                    w1_reduced[k] += beta[i * tensor.dim(1) + j] * v;
                });
            });
        });

        let partial_evals = w1_reduced.clone();
        w1_reduced =
            index_wf(&w1_reduced, pre_fft_shape[2], tensor.dim(2), padded_rows).collect::<Vec<E>>();
        let f_m = w1_reduced.into_mle();

        // Construct the virtual polynomial and run the sumcheck prover
        let f_red = w_red.into_mle();
        let num_vars = f_red.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let expr = [&f_m, &f_red]
            .into_iter()
            .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                acc * expr_builder.lift(Either::Left(p))
            });
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let claims = state.get_mle_flatten_final_evaluations();

        let out_point = state.collect_raw_challenges();
        let (matrix_proofs, matrix_claims, matrix_evaluation_points) =
            prover.delegate_matrix_evaluation(&mut f_middle, &r1, out_point.clone(), false);
        BatchFFTWeightsProof {
            proof,
            claims,
            partial_evals,
            point: out_point,
            matrix_evaluation: (matrix_proofs, matrix_claims),
            matrix_evaluation_points,
        }
    }

    // Prove convolution of a CNN network. This is a convolution between in a 3D matrix X of dimension k_x * n_x * n_x
    // and a 4D filter matrix W of dimension k_w * k_x * n_w * n_w. The output is a 3D matrix Y of dimension k_w * n_x * n_x
    // We want to batch prove the following: Y[i] = iFFT(sum_{j \in [n_x]}(FFT(X[j]) o FFT(W[i][j])).
    #[allow(clippy::too_many_arguments)]
    #[timed::timed_instrument(name = "Prover::prove_convolution_step")]
    fn prove_convolution_step<E, T: Transcript<E>, PCS>(
        &self,
        prover: &mut Prover<E, T, PCS>,
        // last random claim made
        last_claim: &Claim<E>,
        // Struct containing all necessary information
        // to generate a convolution proof
        output: &Tensor<E>,
        unpadded_output_shape: &Shape,
        proving_data: &ConvData<E>,
        id: NodeId,
    ) -> anyhow::Result<Claim<E>>
    where
        E::BaseField: Serialize + DeserializeOwned,
        E: ExtensionField + Serialize + DeserializeOwned,
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
        PCS: PolynomialCommitmentScheme<E> + Send + Sync,
        PCS::ProverParam: Send + Sync,
    {
        let (tensor, pre_fft_shape) = self.filter.as_fft_tensor();
        // First part is proving the clearing of the garbage has been done
        // correctly. For this, we create the clearing garbage tensor and just
        // prove hadamard with the output. This results in two claims: one for
        // the non-cleared tensor and one for the clearing tensor (only 1s and
        // 0s) The non-cleared tensor claim gets passed to the main regular
        // logic of convolution The clearing tensor one gets stored in the proof
        // and will be checked manually by the verifier (CURRENTLY)
        let clearing_tensor = new_clearing_tensor(unpadded_output_shape, output.shape());
        // Take the elements BEFORE bias addition - this is what the rest of the
        // convolution proving step expects.
        //
        // TODO: could trade off less memory by directly recomputing it from
        // conv data with the input shape as well.
        let conv_after_bias = Tensor::new(
            output.shape().clone(),
            proving_data.output_as_element.clone(),
        );
        debug_assert!({
            info!(
                "PROVE: conv_after_bias.shape(): {:?}",
                conv_after_bias.shape()
            );
            info!(
                "PROVE: conv_after_bias.data(): {:?}",
                &conv_after_bias.get_data()[..30]
            );
            info!("PROVE: unpadded_output_shape: {unpadded_output_shape:?}");
            info!("PROVE: output.shape(): {:?}", output.shape());
            let cleared_out = conv_after_bias.to_flatten().mul(&clearing_tensor);
            let fielded: Tensor<E> = cleared_out.to_fields();
            fielded.get_data().to_vec() == output.get_data()
        });
        let clearing_proof = hadamard::prove(
            prover.transcript,
            last_claim,
            &conv_after_bias,
            &clearing_tensor,
        );
        // since v1 is the non cleared tensor, this is what the rest of the convolution proving expects
        let last_claim = Claim::new(
            clearing_proof.random_point().to_vec(),
            clearing_proof.v1_eval(),
        );

        let filter = self;
        assert_eq!(
            filter.fft_filter_size() * filter.kw() * 2,
            proving_data.output.len() * proving_data.output[0].len(),
            "Inconsistent output size"
        );
        assert_eq!(
            (filter.fft_filter_size() * filter.kw()).ilog2() as usize,
            last_claim.point.len(),
            "Inconsistent random point size. Expected : {}, got: {}",
            ((filter.fft_filter_size() * filter.kw()).ilog2()),
            last_claim.point.len()
        );
        let mut r = vec![E::ZERO; last_claim.point.len() + 1];
        let mut bias_point = vec![E::ZERO; filter.kw().ilog2() as usize];
        for (i, item) in r
            .iter_mut()
            .enumerate()
            .take(filter.fft_filter_size().ilog2() as usize)
        {
            *item = E::ONE - last_claim.point[i];
        }
        for i in 0..(filter.kw().ilog2() as usize) {
            r[i + (filter.fft_filter_size().ilog2() as usize) + 1] =
                last_claim.point[i + (filter.fft_filter_size().ilog2() as usize)];
            bias_point[i] = last_claim.point[i + (filter.fft_filter_size().ilog2() as usize)];
        }
        let mut bias_eval = E::ZERO;
        if !bias_point.is_empty() {
            bias_eval = filter.bias.to_field::<E>().into_mle().evaluate(&bias_point);
        } else if filter.bias.data().len() == 1 {
            bias_eval = filter.bias.to_field::<E>()[0];
        }

        debug_assert!({
            let y = proving_data
                .output
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r);
            debug_assert_eq!(last_claim.eval - bias_eval, y, "Error in Conv 1");
            last_claim.eval - bias_eval == y
        });

        let mut temp_t = prover.transcript.clone();
        let BatchFFTProof {
            proof: ifft_proof,
            point: ifft_proof_point,
            claims: ifft_claim,
            matrix_eval: ifft_del_proof,
            delegation_points: ifft_delegation_points,
        } = prover.prove_batch_ifft(r.clone(), &proving_data.prod);

        assert_eq!(
            filter.fft_filter_size().ilog2() as usize + 1,
            ifft_proof_point.len(),
            "Error in ifft sumceck"
        );

        debug_assert!({
            let fft_aux = from_mle_list_dimensions(&[vec![
                (self.fft_filter_size().ilog2() as usize) + 1,
                (self.fft_filter_size().ilog2() as usize) + 1,
            ]]);
            IOPVerifierState::<E>::verify(
                last_claim.eval - bias_eval,
                &ifft_proof.clone(),
                &fft_aux,
                &mut temp_t,
            );
            info!("iFFT Sumcheck Correct");
            true
        });

        // After this point, the verifier holds an evaluation claim of proving_data.prod at P1.randomness[0][i]
        // Let r' = P1.randomness[0][i] and y is the evaluation claim of prod = proving_data.prod
        // What we want to do now is to prove that prod has been correctly computed from X_fft and w (= proving_data.w)
        // In other words we want to show that prod[i] = sum_{j \in [k_x]} x[j] o w[i][j] for each i in [k_w]
        // For this let r1 be the last log(k_w) elements of r and r2 the first log(n_x^2) elements
        // Compute the arrays beta1,beta2 such that beta1[i] = beta(i,r1) and beta2[i] = beta(i,r2)

        let mut r_ifft: Vec<E> = ifft_proof_point.clone();
        for item in r.iter().skip(proving_data.output[0].len().ilog2() as usize) {
            r_ifft.push(*item);
        }

        debug_assert!({
            let eval1 = proving_data
                .prod
                .clone()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .into_mle()
                .evaluate(&r_ifft);
            let eval2 = ifft_claim[0];
            debug_assert_eq!(
                proving_data
                    .prod
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .into_mle()
                    .evaluate(&r_ifft),
                ifft_claim[0],
                "Error in Conv 1"
            );
            eval1 == eval2
        });

        let r1 = &r_ifft[(proving_data.output[0].len().ilog2() as usize)..];
        let r2 = &r_ifft[..(proving_data.output[0].len().ilog2() as usize)];
        let beta1 = compute_betas_eval(r1);
        let beta2 = compute_betas_eval(r2);
        // Given beta1,beta2 observe that :
        // \sum_{i \in [k_w]} beta1[i]prod[i] = \sum_{i \in [k_w]}sum_{j \in [k_x]} x[j] o w[i][j] =
        // = sum_{j \in [k_x]}x[j]o(\sum_{i \in [k_w]}(beta[i]*w[i][j])). We let w_reduced[j] = \sum_{i \in [k_w]}(beta[i]*w[i][j])
        // We have  \sum_{i \in [k_w]} beta1[i]prod[i] = sum_{j \in [k_x]} x[j]o w_{reduced[j]}.
        // So here we compute w_reduced

        let beta_acc = vec![beta2.clone(); filter.kx()].concat();

        // After computing w_reduced, observe that y = \sum_{k \in [n_x^2]} sum_{j \in [k_x]} beta2[k]*x[j][k]*w_reduced[j][k]
        // This is a cubic sumcheck where v1 = [x[0][0],...,x[k_x][n_x^2]], v2 = [w_reduced[0][0],...,w_reduced[k_x][n_x^2]]
        // and v3 = [beta2,..(k_x times)..,beta2]. So, first initialize v3 and then invoke the cubic sumceck.
        let og_filter_size = self.pre_fft_filter_size();
        let mut aggregated_filter = vec![vec![E::ZERO; og_filter_size]; tensor.dim(1)];
        // Compute aggregated_filter using iterators
        // TO DO: PARALLELIZE
        (0..tensor.dim(1)).for_each(|i| {
            (0..tensor.dim(0)).for_each(|j| {
                aggregated_filter[i]
                    .iter_mut()
                    .enumerate()
                    .for_each(|(k, v)| {
                        let index = j * tensor.dim(1) * og_filter_size + i * og_filter_size + k;
                        let v_field: E = tensor[index].to_field();
                        *v += beta1[j] * v_field;
                    });
            });

            aggregated_filter[i] = index_wf(
                &aggregated_filter[i],
                pre_fft_shape[2],
                tensor.dim(2),
                2 * self.fft_filter_size(),
            )
            .collect::<Vec<E>>();

            fft(&mut aggregated_filter[i], false);
        });

        // We need to fix the high variables in place for the filter at r1.
        let f1 = aggregated_filter
            .into_iter()
            .flatten()
            .collect::<Vec<E>>()
            .into_mle();

        let f2 = proving_data
            .input_fft
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_mle();
        let f3 = beta_acc.into_mle();
        let num_vars = f1.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<E>::new(num_threads, num_vars);
        let expr = [&f1, &f2, &f3]
            .into_iter()
            .fold(Expression::Constant(Either::Right(E::ONE)), |acc, p| {
                acc * expr_builder.lift(Either::Left(p))
            });
        let virtual_poly = expr_builder.to_virtual_polys(&[expr], &[]);
        let (hadamard_proof, state) = IOPProverState::<E>::prove(virtual_poly, prover.transcript);

        let hadamard_claims = state.get_mle_flatten_final_evaluations();
        let hadamard_point = state.collect_raw_challenges();

        let point = [hadamard_point.as_slice(), r1].concat();

        // Finally prove the correct computation of the x_fft and get an evaluation claim of the input
        let BatchFFTProof {
            proof: fft_proof,
            claims: fft_claim,
            point: fft_point,
            matrix_eval: fft_del_proof,
            delegation_points: fft_delegation_points,
        } = prover.prove_batch_fft(hadamard_point.clone(), &mut proving_data.input.clone());

        let BatchFFTWeightsProof {
            proof: fft_proof_weights,
            claims: fft_weight_claims,
            point: fft_weight_point,
            partial_evals,
            matrix_evaluation: fft_weights_del_proof,
            matrix_evaluation_points: fft_delegation_weights_points,
        } = self.prove_batch_fft_weights(prover, point.clone());

        let weights_rand: Vec<E> = prover
            .transcript
            .read_challenges((self.pre_fft_filter_size()).ilog2() as usize);
        debug_assert!({
            let mut weights_point = fft_weight_point.clone();
            let mut v_weights = weights_point.pop().unwrap();
            v_weights = (E::ONE - v_weights).inverse();

            let mut r = [
                weights_rand.clone(),
                point[(2 * self.fft_filter_size()).ilog2() as usize..].to_vec(),
            ]
            .concat();
            let y = tensor.to_field::<E>().into_mle().evaluate(&r);
            assert_eq!(
                y,
                partial_evals.clone().into_mle().evaluate(&weights_rand),
                "Error in fft_weights eval"
            );
            let mut indexes = vec![0_usize; self.pre_fft_filter_size()];
            for i in 0..pre_fft_shape[2] {
                for j in 0..pre_fft_shape[2] {
                    indexes[i * pre_fft_shape[2] + j] = i * tensor.dim(2) + j;
                }
            }
            r = weights_point[..(self.fft_filter_size()).ilog2() as usize].to_vec();

            let betas = (0..self.pre_fft_filter_size())
                .map(|i| identity_eval(&r, &to_bits(indexes[i], r.len())))
                .collect::<Vec<_>>();

            let y: E = betas
                .iter()
                .zip(partial_evals.iter())
                .map(|(&beta, &eval)| beta * eval)
                .sum();

            assert_eq!(
                y,
                fft_weight_claims[0] * v_weights,
                "Error in padded weights eval"
            );
            y == fft_weight_claims[0] * v_weights
        });

        let bias_claim = Claim::new(bias_point, bias_eval);
        let filter_claim = Claim::new(
            [
                weights_rand.clone(),
                point[(2 * self.fft_filter_size()).ilog2() as usize..].to_vec(),
            ]
            .concat(),
            partial_evals.clone().into_mle().evaluate(&weights_rand),
        );

        // Add common polynomial commitment claims to the commitment prover
        let common_claims = {
            let mut claims = HashMap::new();
            claims.insert(FILTER_POLY_ID.to_string(), filter_claim);
            claims.insert(BIAS_POLY_ID.to_string(), bias_claim);
            claims
        };
        prover.add_common_claims(id, common_claims);

        prover.push_proof(
            id,
            LayerProof::Convolution(Box::new(ConvProof {
                fft_proof: fft_proof.clone(),
                fft_claims: fft_claim.clone(),
                fft_point: fft_point.clone(),
                fft_proof_weights,
                ifft_proof,
                ifft_point: ifft_proof_point,
                fft_delegation_proof: fft_del_proof.0,
                fft_delegation_proof_weights: fft_weights_del_proof.0,
                ifft_delegation_proof: ifft_del_proof.0,
                ifft_delegation_points,
                hadamard_proof: hadamard_proof.clone(),
                hadamard_point: hadamard_point.clone(),
                ifft_claims: ifft_claim,
                fft_weight_claims,
                fft_weight_point,
                fft_delegation_claims: fft_del_proof.1,
                fft_delegation_points,
                fft_delegation_weights_claims: fft_weights_del_proof.1,
                fft_delegation_weights_points,
                ifft_delegation_claims: ifft_del_proof.1,
                hadamard_clams: hadamard_claims,
                bias_claim: bias_eval,
                partial_evals,
                clearing_proof,
            })),
        );
        let mut input_point = fft_point.clone();
        let mut v = input_point.pop().unwrap();
        v = (E::ONE - v).inverse();
        debug_assert!({
            let mut p = [
                input_point.clone(),
                hadamard_point[((filter.fft_filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat();
            let y = proving_data
                .input
                .clone()
                .into_iter()
                .flat_map(|v| v.into_iter())
                .collect::<Vec<E>>()
                .into_mle()
                .evaluate(&p);
            assert_eq!(y, fft_claim[0] * v, "Error in input eval CONV PROVER");
            for element in p
                .iter_mut()
                .take((filter.fft_filter_size().ilog2()) as usize)
            {
                *element = E::ONE - *element;
            }
            assert_eq!(
                proving_data.real_input.clone().into_mle().evaluate(&p),
                fft_claim[0] * v,
                "Error in real input eval CONV PROVER"
            );
            proving_data.real_input.clone().into_mle().evaluate(&p) == fft_claim[0] * v
        });
        for ip in &mut input_point {
            *ip = E::ONE - *ip;
        }
        let final_claim = Claim {
            point: [
                input_point.clone(),
                hadamard_point[((filter.fft_filter_size() * 2).ilog2() as usize)..].to_vec(),
            ]
            .concat(),
            eval: fft_claim[0] * v,
        };

        Ok(final_claim)
    }
}

struct BatchFFTWeightsProof<E: ExtensionField> {
    proof: sumcheck::structs::IOPProof<E>,
    claims: Vec<E>,
    point: Vec<E>,
    partial_evals: Vec<E>,
    matrix_evaluation: (Vec<sumcheck::structs::IOPProof<E>>, Vec<Vec<E>>),
    matrix_evaluation_points: Vec<Vec<E>>,
}

impl ProveInfo for Convolution<Element> {
    fn step_info<E: ExtensionField>(
        &self,
        id: NodeId,
        mut aux: ContextAux,
    ) -> Result<(LayerCtx<E>, ContextAux)> {
        let (tensor, _) = self.filter.as_fft_tensor();

        let mut filter_shape = tensor.shape().clone();
        filter_shape.remove(1);
        aux.last_output_shape
            .iter_mut()
            .for_each(|shape| *shape = tensor.shape().clone());

        let conv_info = LayerCtx::Convolution(self.conv_context(id));
        let filter_poly = tensor.pad_next_power_of_two().into_data();
        let bias_poly = self.bias.pad_next_power_of_two().into_data();
        aux.model_polys = {
            let mut model_polys = HashMap::new();
            model_polys.insert(FILTER_POLY_ID.to_string(), filter_poly);
            model_polys.insert(BIAS_POLY_ID.to_string(), bias_poly);
            Some(model_polys)
        };
        Ok((conv_info, aux))
    }
}

impl Convolution<f32> {
    fn quantize_from_scalings(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Convolution<Element>>> {
        let model_scaling = ScalingFactor::from_absolute_max(self.max_abs_weight(), None);
        let num_inputs = input_scaling.len();
        ensure!(
            num_inputs == 1,
            "Number of input scaling factor for convolution layer different from 1"
        );
        let input_scaling = &input_scaling[0];
        let bias_scaling = {
            // bias has to be quantized over integers with double bit length
            let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
            let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
            ScalingFactor::from_scale(
                input_scaling.scale() * model_scaling.scale(),
                Some((min_quantized, max_quantized)),
            )
        };
        let quantized_conv = self.quantize(&model_scaling, &bias_scaling);
        let intermediate_bit_size = quantized_conv.output_bitsize();
        let requant = Requant::from_scaling_factors(
            *input_scaling,
            model_scaling,
            output_scaling,
            intermediate_bit_size,
        );

        Ok(QuantizeOutput::new(quantized_conv, vec![output_scaling]).with_requant(requant))
    }
}

impl QuantizeOp for Convolution<f32> {
    type QuantizedOp = Convolution<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len());
        let mut output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        let output_scaling = output_scalings.pop().unwrap();
        self.quantize_from_scalings(input_scaling, output_scaling)
    }
}

impl PadOp for Convolution<Element> {
    fn pad_node(mut self, shape_info: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        let tensor = self.filter.as_pre_fft_tensor();
        ensure!(
            shape_info.shapes.len() == 1,
            "More than 1 input shape found when padding convolution layer"
        );
        let shape_data = shape_info.shapes.first_mut().unwrap();
        shape_data.input_shape_og = safe_conv2d_shape(&shape_data.input_shape_og, tensor.shape())?;
        let weight_shape = tensor.shape();

        // Perform basic sanity checks on the tensor dimensions
        check_filter(weight_shape).context("filter shape test failed:")?;
        ensure!(
            weight_shape[0] == self.bias().shape()[0],
            "Bias length doesn't match filter shape",
        );

        // Make sure that input shape is already padded and is well formed
        ensure!(
            shape_data.input_shape_padded.is_power_of_two(),
            "Input shape for convolution is not padded",
        );
        ensure!(
            shape_data.input_shape_padded.rank() == 3,
            "Input shape for convolution is not 3D",
        );

        // Since we are doing an FFT based conv, we need to pad the last two dimensions of the filter to match the input.
        let filter_shape = tensor.shape().next_power_of_two();
        let (filter_height, filter_width) = (filter_shape[2], filter_shape[3]);
        let (input_height, input_width) = (
            shape_data.input_shape_padded.dim(1),
            shape_data.input_shape_padded.dim(2),
        );

        ensure!(
            filter_height <= input_height && filter_width <= input_width,
            "Filter dimensions in convolution have to be smaller than input dimensions",
        );

        self.prepare_for_fft(&shape_data.input_shape_og);
        let output_shape: Shape = safe_conv2d_shape(&shape_data.input_shape_padded, &filter_shape)?;
        shape_data.input_shape_padded = output_shape.next_power_of_two();
        Ok(self)
    }
}

impl<E, PCS> ProvableOp<E, PCS> for Convolution<Element>
where
    E: ExtensionField,
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type Ctx = proof::ConvCtx;

    fn prove<T: Transcript<E>>(
        &self,
        id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<E>>,
        step_data: &StepData<E, E>,
        prover: &mut Prover<E, T, PCS>,
        store: &mut GenStore,
    ) -> Result<Vec<Claim<E>>> {
        let output_tensor = step_data.output_tensor_at(0, store)?;

        let fft_data = step_data.node_outputs.try_convdata().unwrap();
        let (_, conv_data) = self.fft(&fft_data.input, &fft_data.unpadded_input_shape);

        Ok(vec![self.prove_convolution_step(
            prover,
            last_claims[0],
            &output_tensor,
            &step_data.unpadded_output_shapes[0],
            &conv_data,
            id,
        )?])
    }
}

fn to_bits<E: ExtensionField>(mut num: usize, bitlen: usize) -> Vec<E> {
    let mut bits = vec![E::ZERO; bitlen];
    for bit in bits.iter_mut().take(bitlen) {
        *bit = E::from_canonical_u64((num & 1) as u64);
        num >>= 1;
    }
    bits
}

/// Zero out the padded regions.
///
/// This function iterates over the dimensions that have been increased via padding,
/// and zero out the values in the padded regions, preserving the values in the original
/// space.
fn clear_garbage<T: Number>(output_tensor: &Tensor<T>, unpadded_output_shape: &Shape) -> Tensor<T> {
    let unpadded_output_shape = if unpadded_output_shape.len() == 4 {
        assert_eq!(unpadded_output_shape[0], 1, "Grouping is not supported");
        unpadded_output_shape.slice(1..)
    } else {
        unpadded_output_shape.clone()
    };

    assert_eq!(
        output_tensor.shape().rank(),
        unpadded_output_shape.rank(),
        "The original and padded shapes must have the same rank. original {} padded {}",
        output_tensor.shape().rank(),
        unpadded_output_shape.rank(),
    );
    assert_eq!(
        output_tensor.shape().rank(),
        3,
        "Only rank 3 shapes are supported. got {}",
        output_tensor.shape().rank(),
    );

    let strides = output_tensor.shape().strides();

    let padded_shape = output_tensor.shape();
    let mut data = output_tensor.get_data().to_vec();
    for channel in 0..padded_shape.dim(0) {
        for height in 0..padded_shape.dim(1) {
            for width in 0..padded_shape.dim(2) {
                if !(channel < unpadded_output_shape[0]
                    && height < unpadded_output_shape[1]
                    && width < unpadded_output_shape[2])
                {
                    let index = channel * strides[0] + height * strides[1] + width * strides[2];
                    data[index] = T::default();
                }
            }
        }
    }
    Tensor::new(padded_shape.clone(), data)
}

/// Given an original shapped and its padded counterpart, returns a tensor with ones in
/// the positions matching original and zero in the padded positions.
///
/// The returned tensor can be used to clear out garbage via multiplication.
fn new_clearing_tensor(og_shape: &Shape, padded_shape: &Shape) -> Tensor<Element> {
    let og_shape = if og_shape.len() == 4 {
        assert_eq!(og_shape[0], 1, "Grouping is not supported");
        og_shape.slice(1..)
    } else {
        og_shape.clone()
    };

    assert_eq!(
        padded_shape.rank(),
        og_shape.rank(),
        "The original and padded shapes must have the same rank. original {} padded {}",
        padded_shape.rank(),
        og_shape.rank(),
    );
    assert_eq!(
        padded_shape.rank(),
        3,
        "Only rank 3 shapes are supported. got {}",
        padded_shape.rank(),
    );

    let strides = padded_shape.strides();

    let mut data: Vec<Element> = vec![0; padded_shape.product()];
    for channel in 0..padded_shape.dim(0) {
        for height in 0..padded_shape.dim(1) {
            for width in 0..padded_shape.dim(2) {
                if channel < og_shape.dim(0) && height < og_shape.dim(1) && width < og_shape.dim(2)
                {
                    let index = channel * strides[0] + height * strides[1] + width * strides[2];
                    data[index] = 1;
                }
            }
        }
    }

    Tensor::new(Shape::new(vec![padded_shape.product()]), data)
}

/// Properly pad a filter
/// We use this function so that filter is amenable to FFT based conv2d
/// Usually vec and n are powers of 2
/// Output: [[F[0][0],…,F[0][n_w],0,…,0],[F[1][0],…,F[1][n_w],0,…,0],…]
fn index_wf<E: ExtensionField>(
    w: &[E],
    n_real: usize,
    n: usize,
    output_len: usize,
) -> impl ParallelIterator<Item = E> + use<'_, E> {
    (0..output_len).into_par_iter().map(move |idx| {
        let i = idx / n;
        let j = idx % n;
        if i < n_real && j < n_real {
            w[i * n_real + j]
        } else {
            E::ZERO
        }
    })
}

/// Assumes stride=1, padding=0, and dilation=1
/// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
pub(crate) fn conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Shape {
    let stride = 1usize;
    let padding = 0usize;
    let dilation = 1usize;

    let h_in = if input_shape.len() == 3 {
        input_shape[1]
    } else {
        input_shape[2]
    };
    let kernel = filter_shape[2];
    let h_out = (h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    Shape::new(vec![filter_shape[0], h_out, h_out])
}

/// Similar to conv2d_shape but pads the output shape such that it matches what
/// the padded inference and proving expects
fn padded_conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Shape {
    conv2d_shape(input_shape, filter_shape).next_power_of_two()
}
