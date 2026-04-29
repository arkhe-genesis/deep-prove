use super::*;
use crate::{get_root_of_unity, tensor::KeyedTensor};
use ark_ff::{FftField, Field};
use proptest::prelude::*;
use std::{fmt::Debug, ops::Range};

type F = ark_bn254::Fr;

fn split_garbage(
    fft_output: &Tensor<Element>,
    not_padded_shape: &Shape,
) -> (Vec<Element>, Vec<Element>) {
    let mut not_padded_shape = not_padded_shape.to_vec();
    not_padded_shape.remove(0);
    let mut garbage = Vec::new();
    let mut valid = Vec::new();
    for i in 0..fft_output.shape()[0] {
        for j in 0..fft_output.shape()[1] {
            for k in 0..fft_output.shape()[2] {
                let index = i * fft_output.shape()[1] * fft_output.shape()[2]
                    + j * fft_output.shape()[2]
                    + k;
                let elem = fft_output[index];
                if i < not_padded_shape[0] && j < not_padded_shape[1] && k < not_padded_shape[2] {
                    valid.push(elem);
                } else {
                    garbage.push(elem);
                }
            }
        }
    }
    (valid, garbage)
}

#[test]
fn test_root_of_unity() {
    for i in 2..=F::TWO_ADICITY {
        let c = get_root_of_unity::<F>(i as usize).unwrap();
        assert_eq!(c.pow([1 << i as u64]), F::ONE, "Failed for {i}");
    }
}

/// Checks `conv2d_tensor` and `fft_tensor` have the same result.
///
/// The contents of `conv2d_tensor` must come from a [Tensor::conv2d]
/// operation and have no padding. The `fff_tensor` must be the result of
/// [Tensor::fft_conv]. This utility will skip the garbage values of the fft.
///
/// expected is std conv2d (kw, nx-nw+1, nx-nw+1)
/// fft_tensor is results from fft conv (kw, nx, nx)
pub(crate) fn check_tensor_consistency(
    conv2d_tensor: &Tensor<Element>,
    fft_tensor: &Tensor<Element>,
) {
    assert_eq!(
        fft_tensor.shape().rank(),
        3,
        "FFT tensor should not have batching. shape {:?}",
        fft_tensor.shape(),
    );
    assert_eq!(
        conv2d_tensor.shape().rank(),
        3,
        "Tensor should not have batching. shape {:?}",
        conv2d_tensor.shape(),
    );
    assert_eq!(
        fft_tensor.shape()[1],
        fft_tensor.shape()[2],
        "FFT tensor should have same height and width. shape {:?}",
        fft_tensor.shape(),
    );
    assert!(
        fft_tensor.shape()[2].is_power_of_two(),
        "FFT tensor should have a power-of-two height and width. shape {:?}",
        fft_tensor.shape(),
    );

    let fft_strides = fft_tensor.shape().strides();
    let strides = conv2d_tensor.shape().strides();
    for channel in 0..conv2d_tensor.shape()[0] {
        for height in 0..conv2d_tensor.shape()[1] {
            for width in 0..conv2d_tensor.shape()[2] {
                let expected_pos = channel * strides[0] + height * strides[1] + width * strides[2];
                let fft_pos =
                    channel * fft_strides[0] + height * fft_strides[1] + width * fft_strides[2];

                assert!(
                    conv2d_tensor.data()[expected_pos] == fft_tensor.data()[fft_pos],
                    "Error in tensor consistency. channel {channel} height {height} width {width} got {} expected {}",
                    fft_tensor.data()[fft_pos],
                    conv2d_tensor.data()[expected_pos],
                );
            }
        }
    }
}

#[test]
fn test_conv() {
    for channel in 0..3 {
        for input_size in 2..5 {
            for feature_maps in 0..4 {
                for kernel_size in 1..(input_size - 1) {
                    let filter_size = 1 << kernel_size;
                    let feature_maps = 1 << feature_maps;
                    let input_size = 1 << input_size;
                    let channels = 1 << channel;

                    let filter_shape =
                        Shape::new(vec![feature_maps, channels, filter_size, filter_size]);
                    let padded_input_shape = Shape::new(vec![channels, input_size, input_size]);
                    let input_size = padded_input_shape.numel();

                    let bias = Tensor::<Element>::zeros(Shape::new(vec![feature_maps]));
                    let input =
                        Tensor::new(padded_input_shape.clone(), vec![3; input_size]).unwrap();
                    let mut filter = Filter::new(KeyedTensor::new(
                        "conv_filter",
                        Tensor::random(&filter_shape),
                    ));

                    let expected = input.conv2d(filter.as_pre_fft_tensor(), &bias, 1).unwrap();
                    let expected = expected.squeeze(0).unwrap();

                    filter.prepare_for_fft(&padded_input_shape).unwrap();

                    let (result, _) = filter.fft_conv::<F>(&input, &bias).unwrap();
                    check_tensor_consistency(&expected, &result);
                }
            }
        }
    }
}

#[test]
fn test_clear_garbage() {
    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![1, 1, 2]);
    let tensor = Tensor::new(padded_shape, vec![1, 2]).unwrap();
    assert_eq!(
        clear_garbage(&tensor, &shape).unwrap().data().as_slice(),
        [1, 0],
    );

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![1, 2, 1]);
    let tensor = Tensor::new(padded_shape, vec![1, 2]).unwrap();
    assert_eq!(
        clear_garbage(&tensor, &shape).unwrap().data().as_slice(),
        [1, 0],
    );

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![2, 1, 1]);
    let tensor = Tensor::new(padded_shape, vec![1, 2]).unwrap();
    assert_eq!(
        clear_garbage(&tensor, &shape).unwrap().data().as_slice(),
        [1, 0],
    );

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![1, 2, 2]);
    let tensor = Tensor::new(padded_shape, vec![1, 2, 3, 4]).unwrap();
    assert_eq!(
        clear_garbage(&tensor, &shape).unwrap().data().as_slice(),
        [1, 0, 0, 0],
    );
}

#[test]
fn test_conv2d_shape() {
    let input_shape: Shape = vec![1, 23, 23].into();
    let conv_shape_og: Shape = vec![7, 1, 3, 3].into();
    let output_shape = conv2d_shape(&input_shape, &conv_shape_og);
    assert_eq!(output_shape, vec![7, 21, 21].into());
}

/// Test that check if just taking shapes from input and conv not padded we can manipulate input
/// and filter to run it in padded world with FFT based convolution.
#[test]
fn test_conv_unpadded_to_padded() {
    let input_shape: Shape = vec![1, 23, 23].into();
    let conv_shape_og: Shape = vec![7, 1, 3, 3].into();
    let weight = KeyedTensor::new("conv_filter", Tensor::random(&conv_shape_og));
    let bias = KeyedTensor::new("conv_bias", Tensor::zeros(vec![conv_shape_og[0]].into()));
    let input = Tensor::random(&input_shape);
    let output = input.conv2d(&weight, &bias, 1).unwrap();
    // now try to pad the input and conv and use the fft one
    let padded_input = input.pad_next_power_of_two();
    let fft_conv = Convolution::new(weight.clone(), bias)
        .unwrap()
        .prepared_for_fft(&input_shape)
        .unwrap();
    let (fft_output, conv_data) = fft_conv.fft::<F>(&padded_input).unwrap();
    let (valid, _garbage) = split_garbage(&fft_output, output.shape());
    assert_eq!(
        valid,
        output.data().to_vec(),
        "valid {:?} is not equal to {:?}",
        &valid[..40],
        &output.data()[..40]
    );
    // make sure the shape matches between what we can compute from unpadded and the actual fft output
    let exp_output_shape = conv2d_shape(&input_shape, &conv_shape_og);
    let mut given_output_shape = output.shape().clone();
    given_output_shape.remove(0);
    assert_eq!(given_output_shape, exp_output_shape);

    // make sure we can reconstruct the fft output purely from conv_data since it's needed for proving
    let weight_padded_shape = weight.shape().next_power_of_two();
    let fft_output_shape =
        conv2d_shape(padded_input.shape(), &weight_padded_shape).next_power_of_two();
    assert_eq!(*fft_output.shape(), fft_output_shape);

    let fft_output_data = conv_data.output_as_element;
    let reconstructed_fft_tensor = Tensor::new(fft_output_shape.clone(), fft_output_data).unwrap();
    let hadamard_clearing = new_clearing_tensor(output.shape(), &fft_output_shape).unwrap();
    let hadamard_cleared = reconstructed_fft_tensor
        .to_flatten()
        .mul(&hadamard_clearing);
    assert_eq!(hadamard_cleared.data(), fft_output.data());
}

#[test]
fn convolution_test_simple_element() {
    let channels = 1;
    let filter_size = 2;
    let size = 4;
    let kernels = KeyedTensor::new(
        "conv_filter",
        Tensor::<Element>::new(
            Shape::new(vec![1, channels, filter_size, filter_size]),
            vec![2, 3, 5, 7],
        )
        .unwrap(),
    );
    let input = Tensor::<Element>::new(
        Shape::new(vec![channels, size, size]),
        vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    )
    .unwrap();
    let bias = KeyedTensor::new(
        "conv_bias",
        Tensor::<Element>::new(Shape::new(vec![1]), vec![1]).unwrap(),
    );

    let expected = input.conv2d(&kernels, &bias, 1).unwrap();
    // Remove the leading dimension, the fft only supports 3d tensors.
    let conv2d_result = expected.squeeze(0).unwrap();

    let conv = Convolution::new(kernels, bias)
        .unwrap()
        .prepared_for_fft(input.shape())
        .unwrap();
    let result = conv.evaluate(&[&input.as_wrapped()]).unwrap();
    let fft_result = result.outputs()[0].to_native();

    // Evaluate now returns unpadded output — shapes should match directly.
    assert_eq!(conv2d_result.shape(), fft_result.shape());
    assert_eq!(conv2d_result.data(), fft_result.data());
}

#[test]
fn convolution_test_random_element() {
    let channels = 1;
    let size = 8;
    let filter_size = 4;
    let kernels = KeyedTensor::new(
        "conv_filter",
        Tensor::<Element>::random(&Shape::new(vec![1, channels, filter_size, filter_size])),
    );
    let input = Tensor::<Element>::random(&Shape::new(vec![channels, size, size]));
    let bias = KeyedTensor::new("conv_bias", Tensor::<Element>::random(&Shape::new(vec![1])));

    let expected = input.conv2d(&kernels, &bias, 1).unwrap();
    // Remove the leading dimension, the fft only supports 3d tensors.
    let conv2d_result = expected.squeeze(0).unwrap();

    let conv = Convolution::new(kernels, bias)
        .unwrap()
        .prepared_for_fft(input.shape())
        .unwrap();
    let result = conv.evaluate(&[&input.as_wrapped()]).unwrap();
    let fft_result = result.outputs()[0].to_native();

    // Evaluate now returns unpadded output — shapes should match directly.
    assert_eq!(conv2d_result.shape(), fft_result.shape());
    assert_eq!(conv2d_result.data(), fft_result.data());
}

struct Input<T> {
    kernels: KeyedTensor<T>,
    input: Tensor<T>,
    bias: KeyedTensor<T>,
}

impl<T> Debug for Input<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        fmt.debug_struct("Input")
            .field("input", &format_args!("{:?}", self.input.shape()))
            .field("kernels", &format_args!("{:?}", self.kernels.shape()))
            .field("bias", &format_args!("{:?}", self.bias.shape()))
            .finish()
    }
}

/// FFT convolution is stricter on its input.
///
/// - Only square input arguments, meaning `height == width`.
/// - Only square filters/kernels.
/// - Only 3d input arguments, unlike conv2d 4d is not supported.
/// - Only a single batch is supported by the tensor clearing.
/// - Only strictly smaller filters/kernels than the input
fn input_fft<T: TensorTypeParam>(
    channels: Range<usize>,
    size: Range<usize>,
) -> impl Strategy<Value = Input<T>> {
    (channels, size)
        .prop_filter(
            "Input must be larger than the filter",
            |(_channels, size)| (1 << size) > 4,
        )
        .prop_flat_map(|(channels, size)| {
            let kernels = Tensor::<T>::any(Shape::new(vec![1, 1 << channels, 4, 4]));
            let input = Tensor::<T>::any(Shape::new(vec![1 << channels, 1 << size, 1 << size]));
            let bias = Tensor::<T>::any(Shape::new(vec![1]));
            (kernels, input, bias).prop_map(|(kernels, input, bias)| Input {
                kernels: KeyedTensor::new("fft_conv_filter", kernels),
                input,
                bias: KeyedTensor::new("fft_conv_bias", bias),
            })
        })
}

fn input_conv2d<T: TensorTypeParam>(
    batches: Range<usize>,
    channels: Range<usize>,
    height: Range<usize>,
    width: Range<usize>,
) -> impl Strategy<Value = Input<T>> {
    (batches, channels, height, width).prop_flat_map(|(batches, channels, height, width)| {
        let kernels = Tensor::<T>::any(Shape::new(vec![1 << batches, 1 << channels, 3, 3]));
        let input = Tensor::<T>::any(Shape::new(vec![
            1 << batches,
            1 << channels,
            1 << height,
            1 << width,
        ]));
        let bias = Tensor::<T>::any(Shape::new(vec![1 << batches]));
        (kernels, input, bias).prop_map(|(kernels, input, bias)| Input {
            kernels: KeyedTensor::new("conv_filter", kernels),
            input,
            bias: KeyedTensor::new("conv_bias", bias),
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn convolution_test_single_batch_f32(input in input_conv2d::<f32>(1..2, 1..3, 2..8, 2..8)) {
        let stride = 1;
        let expected = input.input.conv2d(&input.kernels, &input.bias, stride).unwrap();

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone()).unwrap();
        let result = conv.evaluate(&[&input.input.as_wrapped()]).unwrap();

        #[cfg(not(feature = "gpu"))]
        const THRESHOLD: f32 = 1e-3;
        #[cfg(feature = "gpu")]
        const THRESHOLD: f32 = 1e-2;
        result.outputs()[0].get_data().iter().zip(expected.data().iter()).try_for_each(|(left, right)| {
            prop_assert!(
                (left - right).abs() < THRESHOLD,
                "Actual: {left}, Expected: {right}",

            );
            Ok(())
        })?;
    }

    #[test]
    fn convolution_test_multiple_batches_f32(input in input_conv2d::<f32>(1..4, 1..3, 2..8, 2..8)) {
        let stride = 1;
        let expected = input.input.conv2d(&input.kernels, &input.bias, stride).unwrap();

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone()).unwrap();
        let result = conv.evaluate(&[&input.input.as_wrapped()]).unwrap();

        #[cfg(not(feature = "gpu"))]
        const THRESHOLD: f32 = 1e-3;
        #[cfg(feature = "gpu")]
        const THRESHOLD: f32 = 1e-2;
        result.outputs()[0].get_data().iter().zip(expected.data().iter()).try_for_each(|(left, right)| {
            prop_assert!(
                (left - right).abs() < THRESHOLD,
                "Actual: {left}, Expected: {right}",
            );
            Ok(())
        })?;
    }
}

proptest! {
    #[test]
    fn convolution_test_single_batch_element(input in input_fft::<Element>(1..3, 2..7)) {
        let conv2d_result = input.input.conv2d(&input.kernels, &input.bias, 1).unwrap();

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone()).unwrap()
        .prepared_for_fft(input.input.shape()).unwrap();
        let fft_result = conv.evaluate(&[&input.input.as_wrapped()]).unwrap();

        // Evaluate returns unpadded output — shapes should match directly.
        let conv2d_result = conv2d_result.squeeze(0).unwrap();
        let fft_output = fft_result.outputs()[0].to_native();
        prop_assert_eq!(conv2d_result.shape(), fft_output.shape());
        prop_assert_eq!(conv2d_result.data(), fft_output.data());
    }

    #[test]
    fn convolution_test_multiple_batches_element(input in input_fft::<Element>(1..3, 2..7)) {
        let conv2d_result = input.input.conv2d(&input.kernels, &input.bias, 1).unwrap();

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone()).unwrap()
        .prepared_for_fft(input.input.shape()).unwrap();
        let fft_result = conv.evaluate(&[&input.input.as_wrapped()]).unwrap();

        // Evaluate returns unpadded output — shapes should match directly.
        let conv2d_result = conv2d_result.squeeze(0).unwrap();
        let fft_output = fft_result.outputs()[0].to_native();
        prop_assert_eq!(conv2d_result.shape(), fft_output.shape());
        prop_assert_eq!(conv2d_result.data(), fft_output.data());
    }

    #[test]
    fn clear_garbage_and_clearing_tensor_match(channels in 1usize..3, width in 2usize..128, height in 2usize..128) {
        let og_shape = Shape::new(vec![channels, width, height]);
        let padded = Tensor::random(&og_shape.next_power_of_two());

        let clearing_tensor = new_clearing_tensor(&og_shape, padded.shape()).unwrap();
        let cleared_tensor1 = padded.to_flatten().mul(&clearing_tensor);
        let cleared_tensor2 = clear_garbage(&padded, &og_shape).unwrap();

        // The shapes are different, to_flatten creates a 1D tensor, compare only the data.
        assert_eq!(cleared_tensor1.data(), cleared_tensor2.data());
    }
}
