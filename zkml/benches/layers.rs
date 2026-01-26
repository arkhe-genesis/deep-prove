use std::ops::Range;

use zkml::{layers::provable::LayerOut, tensor::TensorTypeParam};

const DATA_SIZE_POWS: Range<i32> = 7..13;

#[derive(Debug, Copy, Clone)]
struct Args {
    pow2: i32,
}

fn default_sizes() -> impl Iterator<Item = Args> {
    DATA_SIZE_POWS.map(|pow2| Args { pow2 })
}

fn sizes(range: Range<i32>) -> impl Iterator<Item = Args> {
    range.map(|pow2| Args { pow2 })
}

/// Gathers the results from the layer.
///
/// For the GPU work this is needed to wait for the computation to finish,
/// otherwise the benchmark is for the time it takes to schedule the work, not
/// to finish it. This has the unfortunate downside of including the time to
/// transfer the data to the GPU.
fn get_results<T>(out: LayerOut<T>) -> Vec<Vec<T>>
where
    T: TensorTypeParam,
{
    out.outputs()
        .iter()
        .map(|wrapped_tensor| wrapped_tensor.get_data())
        .collect()
}

#[divan::bench_group]
mod add_layer {
    use zkml::{
        ScalingFactor, Shape, Tensor,
        layers::{add::Add, provable::Evaluate},
        quantization::Quantize,
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = Tensor::<f32>::random(&shape);
        let input = Tensor::<f32>::random(&shape);
        let result = operand.add(&input);
        let layer = Add::<f32>::new();

        let operand_scaling = ScalingFactor::from_tensor(&operand, None);
        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let result_scaling = ScalingFactor::from_tensor(&result, None);

        let input = WrappedTensor::try_from(input.quantize(&input_scaling)).unwrap();
        let operand = WrappedTensor::try_from(operand.quantize(&operand_scaling)).unwrap();

        let layer = layer
            .quantize(&[operand_scaling, input_scaling], result_scaling)
            .unwrap()
            .quantized_op;

        // warm up
        let out = layer
            .evaluate(&[&input, &operand])
            .expect("Add should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input, &operand])
                .expect("Add should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = WrappedTensor::<f32>::random(&shape);
        let input = WrappedTensor::<f32>::random(&shape);

        let layer = Add::<f32>::new();

        // warm up
        let out = layer
            .evaluate(&[&input, &operand])
            .expect("Add should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input, &operand])
                .expect("Add should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod convolution_layer {
    use std::ops::Range;

    use zkml::{
        Element, Shape, Tensor,
        layers::{convolution::Convolution, provable::Evaluate},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, get_results, sizes};

    // Can not execute convolution layer with size 1<<12 [1]
    //
    // Burn's convolution implementation fails when autotune is used and the
    // input size is 1<<12 or larger. There are two fixes:
    //
    // 1. don't use autotune
    // 2. reduce the input size
    //
    // Unfortunately the first option can not be used, because that completely
    // breaks matmul [2], limiting the input sizes here is a trade off that
    // allows all layers to run with some input.
    //
    // [1]: https://github.com/tracel-ai/burn/issues/3524
    // [2]: https://github.com/tracel-ai/burn/issues/3660
    const F32_SIZES: Range<i32> = 7..12;

    const BATCHES: usize = 1;
    const CHANNELS: usize = 3;

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let kernels = Tensor::<Element>::random(&Shape::new(vec![BATCHES, CHANNELS, 3, 3]));
        let bias = Tensor::<Element>::random(&Shape::new(vec![BATCHES]));

        let input =
            WrappedTensor::<Element>::random(&Shape::new(vec![BATCHES, CHANNELS, size, size]));

        let layer = Convolution::<Element>::new(
            KeyedTensor::new("conv_filter", kernels.clone()),
            KeyedTensor::new("conv_bias", bias.clone()),
        )
        .unwrap()
        .prepared_for_fft(&Shape::from(input.shape()))
        .unwrap();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Convolution should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Convolution should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = sizes(F32_SIZES), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let kernels = Tensor::<f32>::random(&Shape::new(vec![BATCHES, CHANNELS, 3, 3]));
        let bias = Tensor::<f32>::random(&Shape::new(vec![BATCHES]));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![BATCHES, CHANNELS, size, size]));

        let layer = Convolution::<f32>::new(
            KeyedTensor::new("conv_filter", kernels.clone()),
            KeyedTensor::new("conv_bias", bias.clone()),
        )
        .unwrap();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Convolution should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Convolution should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod embeddings_layer {
    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::embeddings::Embeddings},
        number::Number,
        quantization::Quantize,
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let vocab_size = 100;
        let size = 1 << args.pow2;
        let emb = Tensor::<Element>::random(&Shape::new(vec![vocab_size, size]));
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));
        let scaling = ScalingFactor::from_span(
            <f32 as Number>::MIN,
            <f32 as Number>::MAX,
            Some((0, vocab_size as Element)),
        );

        let input = WrappedTensor::try_from(input.quantize(&scaling)).unwrap();

        let layer =
            Embeddings::<Element>::new(KeyedTensor::new("embedding_matrix", emb.clone())).unwrap();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Embeddings should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Embeddings should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let emb = Tensor::<f32>::random(&Shape::new(vec![size, size]));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size]));

        let layer =
            Embeddings::<f32>::new(KeyedTensor::new("embeddings_matrix", emb.clone())).unwrap();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Embeddings should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Embeddings should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod flatten_layer {
    use std::ops::{Range, RangeInclusive};

    use zkml::{
        Element, Shape,
        layers::{flatten::Flatten, provable::Evaluate},
        tensor::WrappedTensor,
    };

    use crate::get_results;

    #[derive(Debug, Copy, Clone)]
    struct Args {
        pow2: i32,
        rank: usize,
    }

    fn args() -> impl Iterator<Item = Args> {
        const DATA_SIZE_POWS: Range<i32> = 4..7;
        const RANKS: RangeInclusive<usize> = 2..=4;

        DATA_SIZE_POWS
            .zip(RANKS)
            .map(|(pow2, rank)| Args { pow2, rank })
    }

    #[divan::bench(args = args(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<Element>::random(&Shape::new([size].repeat(args.rank)));
        let layer = Flatten::default();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Flatten should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Flatten should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = args(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<f32>::random(&Shape::new([size].repeat(args.rank)));
        let layer = Flatten::default();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Flatten should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Flatten should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod gelu_layer {
    use zkml::{
        Shape,
        layers::{activation::GELU, provable::Evaluate},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size]));
        let layer = GELU::<f32>::new();

        // warm up
        let out = layer.evaluate(&[&input]).expect("GeLU should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("GeLU should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod logits_layer {
    use zkml::{
        Element, Shape,
        layers::{provable::Evaluate, transformer::logits::Logits},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let rows = 1 << args.pow2;
        let cols = 16384;
        let shape = Shape::new(vec![rows, cols]);

        let input = WrappedTensor::<Element>::random(&shape);

        let layer = Logits::new_argmax();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Logits should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Logits should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let rows = 1 << args.pow2;
        let cols = 16384;
        let shape = Shape::new(vec![rows, cols]);

        let input = WrappedTensor::<f32>::random(&shape);

        let layer = Logits::new_argmax();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Logits should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Logits should succeed");
            get_results(out)
        });
    }

    #[derive(Debug, Copy, Clone)]
    struct ArgsHighRank {
        d0: usize,
        d1: usize,
        d2: usize,
    }

    fn highrank() -> impl Iterator<Item = ArgsHighRank> {
        [(2, 1024, 1024), (4, 1024, 2048)]
            .into_iter()
            .map(|(d0, d1, d2)| ArgsHighRank { d0, d1, d2 })
    }

    #[divan::bench(args = highrank(), threads = false)]
    fn element_highrank(bencher: divan::Bencher, args: ArgsHighRank) {
        let shape = Shape::new(vec![args.d0, args.d1, args.d2]);

        let input = WrappedTensor::<Element>::random(&shape);

        let layer = Logits::new_argmax();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Logits high rank should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Logits high rank should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = highrank(), threads = false)]
    fn f32_highrank(bencher: divan::Bencher, args: ArgsHighRank) {
        let shape = Shape::new(vec![args.d0, args.d1, args.d2]);

        let input = WrappedTensor::<f32>::random(&shape);

        let layer = Logits::new_argmax();

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Logits high rank should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer
                .evaluate(&[&input])
                .expect("Logits high rank should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod norm_layer {
    use std::ops::Range;

    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::layernorm::LayerNorm},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{DATA_SIZE_POWS, get_results};

    #[derive(Debug, Copy, Clone)]
    struct Args {
        dim0_pow2: i32,
        dim1_pow2: i32,
    }

    const DIM0: Range<i32> = 1..5;
    const EPS: f32 = 1e-5;

    fn args() -> impl Iterator<Item = Args> {
        DIM0.zip(DATA_SIZE_POWS).map(|(dim0_pow2, dim1_pow2)| Args {
            dim0_pow2,
            dim1_pow2,
        })
    }

    #[divan::bench(args = args(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let dim0 = 1 << args.dim0_pow2;
        let dim1 = 1 << args.dim1_pow2;

        let gamma = KeyedTensor::new(
            "norm_gamma",
            Tensor::<Element>::random(&Shape::new(vec![dim1])),
        );
        let beta = KeyedTensor::new(
            "norm_beta",
            Tensor::<Element>::random(&Shape::new(vec![dim1])),
        );
        let layer = LayerNorm::<Element>::new(gamma, beta, EPS).unwrap();

        let input = Tensor::<Element>::random(&Shape::new(vec![dim0, dim1]));

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let input = WrappedTensor::try_from(input).unwrap();
        let (layer, _, _) = layer.quantise(input_scaling, input_scaling).unwrap();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Norm should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Norm should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = args(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let dim0 = 1 << args.dim0_pow2;
        let dim1 = 1 << args.dim1_pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![dim0, dim1]));

        let gamma = KeyedTensor::new("norm_gamma", Tensor::<f32>::random(&Shape::new(vec![dim1])));
        let beta = KeyedTensor::new("norm_beta", Tensor::<f32>::random(&Shape::new(vec![dim1])));
        let layer = LayerNorm::<f32>::new(gamma, beta, EPS).unwrap();

        // warm up
        let out = layer.evaluate(&[&input]).expect("Norm should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Norm should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod positional_absolute_layer {
    use zkml::{
        ScalingFactor, ScalingStrategy, Shape, Tensor,
        layers::{
            provable::{Evaluate, OpInfo, QuantizeOp},
            transformer::positional::Positional,
        },
        padding::PaddingMode,
        quantization::{AbsoluteMax, Quantize},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2; // emb = size
        let context = size * 2;

        let pos = KeyedTensor::new(
            "positional_absolute_mat",
            Tensor::<f32>::random(&Shape::new(vec![context, size])),
        );
        let input_f32 = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_scaling = ScalingFactor::from_tensor(&input_f32, None);
        let input = WrappedTensor::try_from(input_f32.quantize(&input_scaling)).unwrap();

        let base_layer = Positional::<f32>::new_absolute(pos.clone());
        let input_shapes = vec![Shape::new(vec![size, size])];
        let unpadded_output_shapes = base_layer
            .output_shapes(&input_shapes, PaddingMode::NoPadding)
            .expect("positional absolute output shapes");
        let node_id = 0usize.into();
        let output_scalings =
            AbsoluteMax::scaling_factors_for_node(&(), node_id, unpadded_output_shapes.len());

        let layer = base_layer.clone();
        let input_scalings = vec![input_scaling];

        let layer = QuantizeOp::quantize_op::<AbsoluteMax>(
            layer,
            &(),
            node_id,
            &input_scalings,
            &input_shapes,
            &output_scalings,
            &unpadded_output_shapes,
        )
        .expect("quantize positional absolute should succeed")
        .quantized_op;

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Positional absolute should succeed");
        let _ = get_results(out);

        bencher
            .with_inputs(|| {
                layer.reset_cache();
                &layer
            })
            .bench_refs(|layer| {
                let out = layer
                    .evaluate(&[&input])
                    .expect("Positional absolute should succeed");
                get_results(out)
            });
    }
    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2; // emb = size
        let context = size * 2;

        let pos = KeyedTensor::new(
            "positional_absolute_mat",
            Tensor::<f32>::random(&Shape::new(vec![context, size])),
        );

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));
        let layer = Positional::<f32>::new_absolute(pos.clone());

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Positional absolute should succeed");
        let _ = get_results(out);

        bencher
            .with_inputs(|| {
                layer.reset_cache();
                &layer
            })
            .bench_refs(|layer| {
                let out = layer
                    .evaluate(&[&input])
                    .expect("Positional absolute should succeed");
                get_results(out)
            });
    }
}

#[divan::bench_group]
mod positional_rope_layer {
    use std::f32::consts::PI;
    use zkml::{
        ScalingFactor, ScalingStrategy, Shape, Tensor,
        layers::{
            provable::{Evaluate, OpInfo, QuantizeOp},
            transformer::positional::{Positional, RopeLayout},
        },
        padding::PaddingMode,
        quantization::{AbsoluteMax, Quantize},
        tensor::WrappedTensor,
    };

    use crate::{DATA_SIZE_POWS, get_results};

    #[derive(Debug, Copy, Clone)]
    struct RopeArgs {
        pow2: i32,
        layout: RopeLayout,
    }

    fn default_sizes() -> impl Iterator<Item = RopeArgs> {
        DATA_SIZE_POWS.flat_map(|pow2| {
            [
                RopeArgs {
                    pow2,
                    layout: RopeLayout::Adjacent,
                },
                RopeArgs {
                    pow2,
                    layout: RopeLayout::RotateHalf,
                },
            ]
        })
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: RopeArgs) {
        let size: usize = 1 << args.pow2;
        let context = size * 2;
        if size < 2 || !size.is_power_of_two() {
            return;
        }

        let input_f32 = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_scaling = ScalingFactor::from_tensor(&input_f32, None);
        let input = WrappedTensor::try_from(input_f32.quantize(&input_scaling)).unwrap();

        let num_angles = size / 2;
        let angles: Vec<f32> = (0..num_angles)
            .map(|i| ((i as f32) + 1.0) * (PI / (num_angles as f32 + 1.0)))
            .collect();

        let base_layer = Positional::<f32>::new_rope(
            angles,
            "rope_angles".to_string().into(),
            context,
            args.layout,
        )
        .expect("new_rope");
        let input_shapes = vec![Shape::new(vec![size, size])];
        let unpadded_output_shapes = base_layer
            .output_shapes(&input_shapes, PaddingMode::NoPadding)
            .expect("positional rope output shapes");
        let node_id = 0usize.into();
        let output_scalings =
            AbsoluteMax::scaling_factors_for_node(&(), node_id, unpadded_output_shapes.len());

        let input_scalings = vec![input_scaling];
        let layer = QuantizeOp::quantize_op::<AbsoluteMax>(
            base_layer,
            &(),
            node_id,
            &input_scalings,
            &input_shapes,
            &output_scalings,
            &unpadded_output_shapes,
        )
        .expect("quantize positional rope should succeed")
        .quantized_op;

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Positional rope should succeed");
        let _ = get_results(out);

        bencher
            .with_inputs(|| {
                layer.reset_cache();
                &layer
            })
            .bench_refs(|layer| {
                let out = layer
                    .evaluate(&[&input])
                    .expect("Positional rope should succeed");
                get_results(out)
            });
    }
    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: RopeArgs) {
        let size: usize = 1 << args.pow2;
        let context = size * 2;
        if size < 2 || !size.is_power_of_two() {
            return;
        }

        let num_angles = size / 2;
        let angles: Vec<f32> = (0..num_angles)
            .map(|i| ((i as f32) + 1.0) * (PI / (num_angles as f32 + 1.0)))
            .collect();

        let input = WrappedTensor::random(&Shape::new(vec![size, size]));

        let layer = Positional::<f32>::new_rope(
            angles.clone(),
            "rope_angles".to_string().into(),
            context,
            args.layout,
        )
        .expect("new_rope");

        // warm up
        let out = layer
            .evaluate(&[&input])
            .expect("Positional rope should succeed");
        let _ = get_results(out);

        bencher
            .with_inputs(|| {
                layer.reset_cache();
                &layer
            })
            .bench_refs(|layer| {
                let out = layer
                    .evaluate(&[&input])
                    .expect("Positional rope should succeed");
                get_results(out)
            });
    }
}

#[divan::bench_group]
mod softmax_layer {
    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::softmax::Softmax},
        quantization,
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = Tensor::<Element>::random(&Shape::new(vec![size, size]));

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let input = WrappedTensor::try_from(input).unwrap();
        let layer = Softmax::<f32>::new(size)
            .quantise(input_scaling, *quantization::BIT_LEN)
            .expect("Softmax quantise should succeed");

        // warm up
        let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));

        let layer = Softmax::new(size);

        // warm up
        let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod requant_layer {
    use zkml::{
        Element, Shape,
        layers::{provable::Evaluate, requant::Requant},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size]));

        let layer = Requant::from_multiplier(2.0, 8);

        // warm up
        let out = layer.evaluate(&[&input]).expect("Requant should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Requant should succeed");
            get_results(out)
        });
    }
}

#[divan::bench_group]
mod pooling_layer {
    use zkml::{
        Element, Shape,
        layers::{
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::Evaluate,
        },
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes, get_results};

    #[divan::bench(args = default_sizes(), threads = false)]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size, size]));

        let layer = Pooling::Maxpool2D(Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        });

        // warm up
        let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));

        let layer = Pooling::Maxpool2D(Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        });

        // warm up
        let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = layer.evaluate(&[&input]).expect("Softmax should succeed");
            get_results(out)
        });
    }
}
#[divan::bench_group]
mod einsum_layer {
    use std::ops::Range;

    use zkml::{
        Element, Shape, Tensor,
        layers::{einsum::EinSum, provable::Evaluate},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, get_results, sizes};

    // XXX: beyond this point benchmarks for elements are too slow, see matmul
    // benches for measurements.
    //
    //                   fastest | slowest | median  | mean    | samples │ iters
    // Args { pow2: 11 } 2.36 m  | 2.36 m  | 2.36 m  | 2.36 m  | 1       | 1
    const ELEMENT_SIZES: Range<i32> = 7..10;
    const CONCATS: usize = 8;

    #[divan::bench(args = sizes(ELEMENT_SIZES), threads = false)]
    fn einsum_qkv_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let q = KeyedTensor::new("qkv_weight.q", Tensor::random(&vec![size, size].into()));
        let q_bias = KeyedTensor::new("qkv_bias.q", Tensor::random(&vec![size].into()));
        let k = KeyedTensor::new("qkv_weight.k", Tensor::random(&vec![size, size].into()));
        let k_bias = KeyedTensor::new("qkv_bias.k", Tensor::random(&vec![size].into()));
        let v = KeyedTensor::new("qkv_weight.v", Tensor::random(&vec![size, size].into()));
        let v_bias = KeyedTensor::new("qkv_bias.v", Tensor::random(&vec![size].into()));

        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size, size]));

        let einsum_layer = EinSum::<Element>::new(
            "X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)".to_string(),
            vec![Some(q), Some(k), Some(v)],
            vec![Some(q_bias), Some(k_bias), Some(v_bias)],
        )
        .unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&input])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&input])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn einsum_qkv_f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let q = KeyedTensor::new(
            "qkv_weight.q",
            Tensor::<f32>::random(&vec![size, size].into()),
        );
        let q_bias = KeyedTensor::new("qkv_bias.q", Tensor::random(&vec![size].into()));
        let k = KeyedTensor::new("qkv_weight.k", Tensor::random(&vec![size, size].into()));
        let k_bias = KeyedTensor::new("qkv_bias.k", Tensor::random(&vec![size].into()));
        let v = KeyedTensor::new("qkv_weight.v", Tensor::random(&vec![size, size].into()));
        let v_bias = KeyedTensor::new("qkv_bias.v", Tensor::random(&vec![size].into()));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));

        let einsum_layer = EinSum::<f32>::new(
            "X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)".to_string(),
            vec![Some(q), Some(k), Some(v)],
            vec![Some(q_bias), Some(k_bias), Some(v_bias)],
        )
        .unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&input])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&input])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = sizes(ELEMENT_SIZES), threads = false)]
    fn einsum_concat_matmul_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = WrappedTensor::<Element>::random(&shape);
        let right = WrappedTensor::<Element>::random(&shape);
        let einsum_layer =
            EinSum::<Element>::new("A(kij)@B(jil)->C(ikl)".to_string(), vec![None], vec![None])
                .unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&left, &right])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&left, &right])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn einsum_concat_matmul_f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = WrappedTensor::<f32>::random(&shape);
        let right = WrappedTensor::<f32>::random(&shape);
        let einsum_layer =
            EinSum::<f32>::new("A(kij)@B(jil)->C(ikl)".to_string(), vec![None], vec![None])
                .unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&left, &right])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&left, &right])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = sizes(ELEMENT_SIZES), threads = false)]
    fn einsum_matmul_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = WrappedTensor::<Element>::random(&vec![size, size].into());
        let right = WrappedTensor::<Element>::random(&vec![size, size].into());

        let einsum_layer =
            EinSum::<Element>::new("A(ij)@B(kj)->C(ik)".to_string(), vec![None], vec![None])
                .unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&left, &right])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&left, &right])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }

    #[divan::bench(args = default_sizes(), threads = false)]
    fn einsum_matmul_f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = WrappedTensor::<f32>::random(&vec![size, size].into());
        let right = WrappedTensor::<f32>::random(&vec![size, size].into());

        let einsum_layer =
            EinSum::<f32>::new("A(ij)@B(kj)->C(ik)".to_string(), vec![None], vec![None]).unwrap();

        // warm up
        let out = einsum_layer
            .evaluate(&[&left, &right])
            .expect("EinSum should succeed");
        let _ = get_results(out);

        bencher.bench(|| {
            let out = einsum_layer
                .evaluate(&[&left, &right])
                .expect("EinSum should succeed");
            get_results(out)
        });
    }
}

fn main() {
    divan::main();
}
