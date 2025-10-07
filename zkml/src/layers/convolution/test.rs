use super::*;
use crate::{
    layers::{
        activation::Activation,
        dense::Dense,
        pooling::{Maxpool2D, Pooling, maxpool2d_shape},
        provable::evaluate_layer,
    },
    tensor::{KeyedTensor, check_tensor_consistency},
};
use ff_ext::GoldilocksExt2;
use proptest::prelude::*;
use std::{fmt::Debug, ops::Range};

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
                    let input = Tensor::new(padded_input_shape.clone(), vec![3; input_size]);
                    let mut filter = Filter::new(KeyedTensor::new(
                        "conv_filter",
                        Tensor::random(&filter_shape),
                    ));

                    let expected = input.conv2d(filter.as_pre_fft_tensor(), &bias, 1);
                    let expected = expected.squeeze(0);

                    filter.prepare_for_fft(&padded_input_shape);

                    let (result, _) = filter.fft_conv::<GoldilocksExt2>(&input, &bias);
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
    let tensor = Tensor::new(padded_shape, vec![1, 2]);
    assert_eq!(clear_garbage(&tensor, &shape).data(), [1, 0]);

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![1, 2, 1]);
    let tensor = Tensor::new(padded_shape, vec![1, 2]);
    assert_eq!(clear_garbage(&tensor, &shape).data(), [1, 0]);

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![2, 1, 1]);
    let tensor = Tensor::new(padded_shape, vec![1, 2]);
    assert_eq!(clear_garbage(&tensor, &shape).data(), [1, 0]);

    let shape = Shape::new(vec![1, 1, 1]);
    let padded_shape = Shape::new(vec![1, 2, 2]);
    let tensor = Tensor::new(padded_shape, vec![1, 2, 3, 4]);
    assert_eq!(clear_garbage(&tensor, &shape).data(), [1, 0, 0, 0]);
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
    let output = input.conv2d(&weight, &bias, 1);
    // now try to pad the input and conv and use the fft one
    let padded_input = input.pad_next_power_of_two();
    let fft_conv = Convolution::new(weight.clone(), bias).prepared_for_fft(&input_shape);
    let (fft_output, conv_data) = fft_conv.fft::<GoldilocksExt2>(&padded_input, &input_shape);
    let (valid, _garbage) = split_garbage(&fft_output, output.shape());
    assert_eq!(
        valid,
        output.get_data().to_vec(),
        "valid {:?} is not equal to {:?}",
        &valid[..40],
        &output.get_data()[..40]
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
    let reconstructed_fft_tensor = Tensor::new(fft_output_shape.clone(), fft_output_data);
    let hadamard_clearing = new_clearing_tensor(output.shape(), &fft_output_shape);
    let hadamard_cleared = reconstructed_fft_tensor
        .to_flatten()
        .mul(&hadamard_clearing);
    assert_eq!(hadamard_cleared.get_data(), fft_output.get_data());
}

#[test]
fn test_conv_padding_garbage() {
    let input_shape: Shape = vec![1, 23, 23].into();
    let conv_shape_og: Shape = vec![7, 1, 3, 3].into();

    // weight of the filter
    let w1 = KeyedTensor::new("conv_filter", Tensor::random(&conv_shape_og));
    let bias1 = KeyedTensor::new("conv_bias", Tensor::zeros(vec![conv_shape_og[0]].into()));
    // creation of the padded and fft'd convolution
    let fft_conv = Convolution::new(w1.clone(), bias1.clone()).prepared_for_fft(&input_shape);
    let input = Tensor::random(&input_shape);
    let padded_input = input.pad_next_power_of_two();
    let (fft_output, _): (Tensor<Element>, ConvData<_>) =
        fft_conv.fft::<GoldilocksExt2>(&padded_input, &input_shape);
    // just normal convolution
    let normal_output = input.conv2d(&w1, &bias1, 1);

    // Flatten for the dense layer
    let flat_fft_output = fft_output.to_flatten();
    let flat_normal_output = normal_output.to_flatten();
    // Check that the garbage and valid parts are correct
    let (valid, garbage) = split_garbage(&fft_output, normal_output.shape());
    assert!(valid.len() == flat_normal_output.get_data().len());
    assert_eq!(valid, flat_normal_output.get_data().to_vec());
    assert!(!garbage.is_empty());
    // NOTE: a bit of a hack to recreate but the functione xpects the real conv shape not the flattened one
    let (valid, garbage) = split_garbage(
        &Tensor::new(
            fft_output.shape().clone(),
            flat_fft_output.get_data().to_vec(),
        ),
        normal_output.shape(),
    );
    // at this point the garbage should be all zeros and the valid should be the same as the non fft output as before
    assert!(garbage.iter().all(|x| *x == 0));
    assert!(valid == flat_normal_output.get_data().to_vec());

    // dense output to REMOVE garbage - even tho it is only zero now we still need to remove it to get the right shape
    // dense layer should have exactly the same number of columns as the flat normal output
    let ncols = flat_normal_output.shape()[0];
    let nrows = 10;
    let dense_shape = vec![nrows, ncols];
    let dense = Dense::new(
        KeyedTensor::new(
            "dense_weight",
            Tensor::new(
                dense_shape.clone().into(),
                vec![1; dense_shape.iter().product()],
            ),
        ),
        KeyedTensor::new("dense_bias", Tensor::zeros(vec![dense_shape[0]].into())),
    );
    // create the padded version:
    // take the "conv2d"input shape
    let conv_input_shape = conv2d_shape(&input_shape, w1.shape());
    let conv_input_shape_padded = conv_input_shape.next_power_of_two();
    let dense_shape_padded = vec![
        nrows.next_power_of_two(),
        flat_fft_output.shape()[0].next_power_of_two(),
    ];
    let mut padded_dense = dense.clone();
    padded_dense.matrix = padded_dense.matrix.map_tensor(|t| {
        t.pad_matrix_to_ignore_garbage(
            &conv_input_shape,
            &conv_input_shape_padded,
            &dense_shape_padded.into(),
        )
    });
    let padded_nrows = padded_dense.nrows();
    padded_dense.bias = padded_dense
        .bias
        .map(|b| b.map_tensor(|t| t.pad_1d(padded_nrows)));
    let no_garbage_fft_output =
        evaluate_layer::<GoldilocksExt2, _, _>(&padded_dense, &[&flat_fft_output], None)
            .unwrap()
            .outputs()[0]
            .clone();
    let no_garbage_normal_output =
        evaluate_layer::<GoldilocksExt2, _, _>(&dense, &[&flat_normal_output], None)
            .unwrap()
            .outputs()[0]
            .clone();
    let max_rows = dense.nrows();
    assert_eq!(
        &no_garbage_fft_output.get_data()[..max_rows],
        no_garbage_normal_output.get_data()
    );
    assert!(
        no_garbage_fft_output.get_data()[max_rows..]
            .iter()
            .all(|x| *x == 0)
    );
}

#[test]
pub fn test_conv_fft_vs_naive() -> anyhow::Result<()> {
    let n_w = 1 << 2;
    let k_w = 1 << 0;
    let k_x = 1 << 0;

    let mut input_shape_og: Shape = vec![k_x, 256, 256].into();
    let mut input_shape_padded: Shape = input_shape_og.next_power_of_two();
    let filter = KeyedTensor::new(
        "conv_filter",
        Tensor::random(&vec![k_w, k_x, n_w, n_w].into()),
    );
    let bias = KeyedTensor::new("conv_bias", Tensor::random(&vec![k_w].into()));
    let input = Tensor::random(&input_shape_og);

    let output = input.conv2d(&filter, &bias, 1);
    let dims = filter.shape();
    let fft_conv = Convolution::new(filter.clone(), bias).prepared_for_fft(&input_shape_og);
    let mut fft_input = input.clone();
    fft_input.pad_to_shape(input_shape_padded.clone());
    let (fft_output, _proving_data) = fft_conv.fft::<GoldilocksExt2>(&fft_input, &input_shape_og);

    input_shape_og = conv2d_shape(&input_shape_og, filter.shape());
    input_shape_padded = conv2d_shape(&input_shape_padded, dims).next_power_of_two();

    // add a RELU layer
    let relu = Activation::new_relu();
    let output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &[&output], None)
        .unwrap()
        .outputs()[0]
        .clone();
    let fft_output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &[&fft_output], None)
        .unwrap()
        .outputs()[0]
        .clone();

    // make a pooled output
    let pool = Pooling::Maxpool2D(Maxpool2D::default());
    let output = pool.op(&output);
    let fft_output = pool.op(&fft_output);
    input_shape_og = maxpool2d_shape(&input_shape_og);
    input_shape_padded = maxpool2d_shape(&input_shape_padded);

    // again another conv
    let filter = KeyedTensor::new(
        "conv2_filter",
        Tensor::random(&vec![k_w, k_x, n_w, n_w].into()),
    );
    let bias = KeyedTensor::new("conv2_bias", Tensor::random(&vec![k_w].into()));
    println!("2AND CONV: filter.shape() : {:?}", filter.shape());
    println!("2AND CONV: bias.shape() : {:?}", bias.shape());
    println!("2AND CONV: input.shape() : {:?}", output.shape());
    let output = output.conv2d(&filter, &bias, 1);
    let dims = filter.shape();
    let fft_conv = Convolution::new(filter.clone(), bias).prepared_for_fft(&input_shape_padded);
    let mut fft_input = fft_output;
    fft_input.pad_to_shape(input_shape_padded.clone());
    let (fft_output, _proving_data) = fft_conv.fft::<GoldilocksExt2>(&fft_input, &input_shape_og);

    input_shape_og = conv2d_shape(&input_shape_og, filter.shape());
    input_shape_padded = conv2d_shape(&input_shape_padded, dims).next_power_of_two();

    // Add another RELU
    let relu = Activation::new_relu();
    let output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &[&output], None)
        .unwrap()
        .outputs()[0]
        .clone();
    let fft_output = evaluate_layer::<GoldilocksExt2, _, _>(&relu, &[&fft_output], None)
        .unwrap()
        .outputs()[0]
        .clone();

    // make a pooled output
    let pool = Pooling::Maxpool2D(Maxpool2D::default());
    let output = pool.op(&output);
    let fft_output = pool.op(&fft_output);
    input_shape_og = maxpool2d_shape(&input_shape_og);
    input_shape_padded = maxpool2d_shape(&input_shape_padded);

    // now dense layer - first there is a "reshape" that flattens the input
    let ignore_garbage_pad = (input_shape_og.clone(), input_shape_padded.clone());
    input_shape_og = vec![input_shape_og.iter().product()].into();
    input_shape_padded = vec![input_shape_padded.iter().product()].into();

    let nrows = 10;
    let ncols = input_shape_og[0];
    let weight = Tensor::random(&vec![nrows, ncols].into());
    let bias = Tensor::random(&vec![nrows].into());
    let mut new_cols = ncols.next_power_of_two();
    let new_rows = nrows.next_power_of_two();
    if new_cols < input_shape_padded[0] {
        // must make sure that we can apply the input to this padded dense
        new_cols = input_shape_padded[0];
    }
    let conv_shape_og = ignore_garbage_pad.0.clone();
    let conv_shape_pad = ignore_garbage_pad.1.clone();
    let dense = Dense::new(
        KeyedTensor::new("dense_weight", weight.clone()),
        KeyedTensor::new("dense_bias", bias.clone()),
    );
    let dense_output = evaluate_layer::<GoldilocksExt2, _, _>(&dense, &[&output], None)
        .unwrap()
        .outputs()[0]
        .clone();

    let fft_weight = weight.pad_matrix_to_ignore_garbage(
        &conv_shape_og,
        &conv_shape_pad,
        &vec![new_rows, new_cols].into(),
    );
    let fft_bias = bias.clone().pad_1d(new_rows);
    let fft_dense = Dense::new(
        KeyedTensor::new("fft_dense_weight", fft_weight.clone()),
        KeyedTensor::new("fft_dense_bias", fft_bias.clone()),
    );
    println!("-- new_rows : {new_rows}, new_cols : {new_cols}");
    println!("weight.shape() : {:?}", weight.shape());
    println!("bias.shape() : {:?}", bias.shape());
    println!("fft_input.shape() : {:?}", fft_output.shape());
    println!("fft_weight.shape() : {:?}", fft_weight.shape());
    println!("fft_bias.shape() : {:?}", fft_bias.shape());
    println!(
        "output shape : {:?} - product {}",
        output.shape(),
        output.shape().iter().product::<usize>()
    );
    let fft_dense_output = evaluate_layer::<GoldilocksExt2, _, _>(&fft_dense, &[&fft_output], None)
        .unwrap()
        .outputs()[0]
        .clone();
    assert_eq!(
        dense_output.get_data()[..weight.nrows_2d()],
        fft_dense_output.get_data()[..weight.nrows_2d()]
    );
    Ok(())
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
        ),
    );
    let input = Tensor::<Element>::new(
        Shape::new(vec![channels, size, size]),
        vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    );
    let bias = KeyedTensor::new(
        "conv_bias",
        Tensor::<Element>::new(Shape::new(vec![1]), vec![1]),
    );

    let expected = input.conv2d(&kernels, &bias, 1);
    // Remove the leading dimension, the fft only supports 3d tensors.
    let mut conv2d_result = expected.squeeze(0);

    let conv = Convolution::new(kernels, bias).prepared_for_fft(input.shape());
    let result = conv
        .evaluate::<GoldilocksExt2>(&[&input], &[input.shape().clone()])
        .unwrap();
    let fft_result = result.outputs()[0];

    check_tensor_consistency(&conv2d_result, fft_result);

    // Pad the conv2d result to match the fft padded shape with the extra values set to 0.
    conv2d_result.pad_to_shape(fft_result.shape().clone());

    assert_eq!(conv2d_result.get_data(), fft_result.get_data());
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

    let expected = input.conv2d(&kernels, &bias, 1);
    // Remove the leading dimension, the fft only supports 3d tensors.
    let mut conv2d_result = expected.squeeze(0);

    let conv = Convolution::new(kernels, bias).prepared_for_fft(input.shape());
    let result = conv
        .evaluate::<GoldilocksExt2>(&[&input], &[input.shape().clone()])
        .unwrap();
    let fft_result = result.outputs()[0];

    check_tensor_consistency(&conv2d_result, fft_result);

    // Pad the conv2d result to match the fft padded shape with the extra values set to 0.
    conv2d_result.pad_to_shape(fft_result.shape().clone());

    assert_eq!(conv2d_result.get_data(), fft_result.get_data());
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
fn input_fft<T: Number>(
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

fn input_conv2d<T: Number>(
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
    #[test]
    fn convolution_test_single_batch_f32(input in input_conv2d::<f32>(1..2, 1..3, 2..8, 2..8)) {
        let stride = 1;
        let expected = input.input.conv2d(&input.kernels, &input.bias, stride);

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone());
        let result = conv.evaluate::<GoldilocksExt2>(&[&input.input], &[]).unwrap();

        result.outputs()[0].get_data().iter().zip(expected.get_data().iter()).try_for_each(|(left, right)| {
            prop_assert!(
                (left - right).abs() < 1e-3,
                "Actual: {left}, Expected: {right}",

            );
            Ok(())
        })?;
    }

    #[test]
    fn convolution_test_multiple_batches_f32(input in input_conv2d::<f32>(1..4, 1..3, 2..8, 2..8)) {
        let stride = 1;
        let expected = input.input.conv2d(&input.kernels, &input.bias, stride);

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone());
        let result = conv.evaluate::<GoldilocksExt2>(&[&input.input], &[]).unwrap();

        result.outputs()[0].get_data().iter().zip(expected.get_data().iter()).try_for_each(|(left, right)| {
            prop_assert!(
                (left - right).abs() < 1e-3,
                "Actual: {left}, Expected: {right}",
            );
            Ok(())
        })?;
    }

    #[test]
    fn convolution_test_single_batch_element(input in input_fft::<Element>(1..3, 2..7)) {
        let conv2d_result = input.input.conv2d(&input.kernels, &input.bias, 1);

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone())
        .prepared_for_fft(input.input.shape());
        let fft_result = conv.evaluate::<GoldilocksExt2>(&[&input.input], &[input.input.shape().clone()]).unwrap();

        // Remove the leading dimension, the fft only supports 3d tensors.
        let conv2d_result = conv2d_result.squeeze(0);
        check_tensor_consistency(&conv2d_result, fft_result.outputs()[0]);
    }

    #[test]
    fn convolution_test_multiple_batches_element(input in input_fft::<Element>(1..3, 2..7)) {
        let conv2d_result = input.input.conv2d(&input.kernels, &input.bias, 1);

        let conv = Convolution::new(input.kernels.clone(), input.bias.clone())
        .prepared_for_fft(input.input.shape());
        let fft_result = conv.evaluate::<GoldilocksExt2>(&[&input.input], &[input.input.shape().clone()]).unwrap();

        // Remove the leading dimension, the fft only supports 3d tensors.
        let conv2d_result = conv2d_result.squeeze(0);
        check_tensor_consistency(&conv2d_result, fft_result.outputs()[0]);
    }

    #[test]
    fn clear_garbage_and_clearing_tensor_match(channels in 1usize..3, width in 2usize..128, height in 2usize..128) {
        let og_shape = Shape::new(vec![channels, width, height]);
        let padded = Tensor::random(&og_shape.next_power_of_two());

        let clearing_tensor = new_clearing_tensor(&og_shape, padded.shape());
        let cleared_tensor1 = padded.to_flatten().mul(&clearing_tensor);
        let cleared_tensor2 = clear_garbage(&padded, &og_shape);
        assert_eq!(cleared_tensor1.get_data(), cleared_tensor2.get_data());
    }
}
