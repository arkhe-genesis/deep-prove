use std::ops::Range;

const DATA_SIZE_POWS: Range<i32> = 7..14;

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

#[divan::bench_group]
mod add_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        ScalingFactor, Shape, Tensor,
        layers::{add::Add, provable::Evaluate},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = Tensor::<f32>::random(&shape);
        let input = Tensor::<f32>::random(&shape);
        let result = operand.add(&input);
        let layer = Add::<f32>::new_with(
            KeyedTensor::new("add_operand", operand.clone()),
            shape.clone(),
        );

        let operand_scaling = ScalingFactor::from_tensor(&operand, None);
        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let result_scaling = ScalingFactor::from_tensor(&result, None);

        let input = WrappedTensor::try_from(&input.to_quantized(&input_scaling)).unwrap();
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = layer
            .quantize(&[operand_scaling, input_scaling], result_scaling)
            .unwrap()
            .quantized_op;

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Add should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = Tensor::<f32>::random(&shape);

        let input = WrappedTensor::<f32>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Add::<f32>::new_with(
            KeyedTensor::new("add_operand", operand.clone()),
            shape.clone(),
        );

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Add should succeed")
        });
    }
}

#[divan::bench_group]
mod dense_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape, Tensor,
        layers::{dense::Dense, provable::Evaluate},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let matrix = Tensor::<Element>::random(&Shape::new(vec![size, size]));
        let bias = Tensor::<Element>::random(&Shape::new(vec![size]));

        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Dense::<Element>::new(
            KeyedTensor::new("dense_weight", matrix.clone()),
            KeyedTensor::new("dense_bias", bias.clone()),
        );

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Dense should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let matrix = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let bias = Tensor::<f32>::random(&Shape::new(vec![size]));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Dense::<f32>::new(
            KeyedTensor::new("dense_weight", matrix.clone()),
            KeyedTensor::new("dense_bias", bias.clone()),
        );

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Dense should succeed")
        });
    }
}

#[divan::bench_group]
mod convolution_layer {
    use core::slice;
    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape, Tensor,
        layers::{convolution::Convolution, provable::Evaluate},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, sizes};

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

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let kernels = Tensor::<Element>::random(&Shape::new(vec![BATCHES, CHANNELS, 3, 3]));
        let bias = Tensor::<Element>::random(&Shape::new(vec![BATCHES]));

        let input =
            WrappedTensor::<Element>::random(&Shape::new(vec![BATCHES, CHANNELS, size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Convolution::<Element>::new(
            KeyedTensor::new("conv_filter", kernels.clone()),
            KeyedTensor::new("conv_bias", bias.clone()),
        )
        .prepared_for_fft(&Shape::from(input.shape()));

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Convolution should succeed")
        });
    }

    #[divan::bench(args = sizes(F32_SIZES))]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let kernels = Tensor::<f32>::random(&Shape::new(vec![BATCHES, CHANNELS, 3, 3]));
        let bias = Tensor::<f32>::random(&Shape::new(vec![BATCHES]));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![BATCHES, CHANNELS, size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Convolution::<f32>::new(
            KeyedTensor::new("conv_filter", kernels.clone()),
            KeyedTensor::new("conv_bias", bias.clone()),
        );

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Convolution should succeed")
        });
    }
}

#[divan::bench_group]
mod embeddings_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::embeddings::Embeddings},
        number::Number,
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
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

        let input = WrappedTensor::try_from(&input.to_quantized(&scaling)).unwrap();
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer =
            Embeddings::<Element>::new(KeyedTensor::new("embedding_matrix", emb.clone())).unwrap();
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Embeddings should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let emb = Tensor::<f32>::random(&Shape::new(vec![size, size]));

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer =
            Embeddings::<f32>::new(KeyedTensor::new("embeddings_matrix", emb.clone())).unwrap();
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Embeddings should succeed")
        });
    }
}

#[divan::bench_group]
mod flatten_layer {
    use core::slice;
    use std::ops::{Range, RangeInclusive};

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{flatten::Flatten, provable::Evaluate},
        tensor::WrappedTensor,
    };

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

    #[divan::bench(args = args())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<Element>::random(&Shape::new([size].repeat(args.rank)));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);
        let layer = Flatten;
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Flatten should succeed")
        });
    }

    #[divan::bench(args = args())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<f32>::random(&Shape::new([size].repeat(args.rank)));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);
        let layer = Flatten;
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Flatten should succeed")
        });
    }
}

#[divan::bench_group]
mod gelu_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Shape,
        layers::{activation::GELU, provable::Evaluate},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);
        let layer = GELU::<f32>::new();
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("GeLU should succeed")
        });
    }
}

#[divan::bench_group]
mod matmul_layer {

    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape, Tensor,
        layers::{
            matrix_mul::{self, MatMul},
            provable::Evaluate,
        },
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, sizes};

    // XXX: beyond these sizes the benchmarks for elements are extremely slow.
    //
    //                   fastest | slowest | median  | mean    | samples │ iters
    // Args { pow2: 11 } 10.26 s | 10.87 s | 10.44 s | 10.52 s | 3       | 3
    // Args { pow2: 12 } 1.357 m | 1.357 m | 1.357 m | 1.357 m | 1       | 1
    // Args { pow2: 13 }  1.67 h |  1.67 h |  1.67 h |  1.67 h | 1       | 1
    const ELEMENT_SIZES: Range<i32> = 7..10;

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = WrappedTensor::<Element>::random(&vec![size, size].into());
        let right = WrappedTensor::<Element>::random(&vec![size, size].into());
        let bias = KeyedTensor::new("matmul_bias", Tensor::<Element>::random(&vec![size].into()));
        let config = matrix_mul::Config::TransposeB;

        let layer = MatMul::<Element>::new_with_config(
            matrix_mul::OperandMatrix::Input,
            matrix_mul::OperandMatrix::Input,
            Some(bias),
            config,
        )
        .unwrap();
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[Shape::from(left.shape()), Shape::from(right.shape())],
                )
                .expect("MatMul should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = WrappedTensor::<f32>::random(&vec![size, size].into());
        let right = WrappedTensor::<f32>::random(&vec![size, size].into());
        let bias = KeyedTensor::new("matmul_bias", Tensor::<f32>::random(&vec![size].into()));
        let config = matrix_mul::Config::TransposeB;

        let layer = MatMul::<f32>::new_with_config(
            matrix_mul::OperandMatrix::Input,
            matrix_mul::OperandMatrix::Input,
            Some(bias),
            config,
        )
        .unwrap();

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[Shape::from(left.shape()), Shape::from(right.shape())],
                )
                .expect("MatMul should succeed")
        });
    }
}

#[divan::bench_group]
mod concat_matmul_layer {

    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{
            concat_matmul::{ConcatMatMul, InputMatrixDimensions, Permutation},
            provable::Evaluate,
        },
        tensor::WrappedTensor,
    };

    use crate::{Args, sizes};

    // XXX: beyond this point benchmarks for elements are too slow, see matmul
    // benches for measurements.
    //
    // Args { pow2: 11 } 3.206 m | 3.206 m | 3.206 m | 3.206 m | 1 | 1
    const ELEMENT_SIZES: Range<i32> = 7..10;
    const CONCATS: usize = 8;

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left_perm = InputMatrixDimensions::new(1, 2, 0);
        let right_perm = InputMatrixDimensions::new(1, 0, 2);
        let out_perm = Permutation::new(vec![2, 1, 0]);

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = WrappedTensor::<Element>::random(&shape);
        let right = WrappedTensor::<Element>::random(&shape);

        let layer = ConcatMatMul::new_with_permute(left_perm, right_perm, out_perm);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[Shape::from(left.shape()), Shape::from(right.shape())],
                )
                .expect("ConcatMatMul should succeed")
        });
    }

    // NOTE Upper limit set to 2^12 as 2^13 would make a tensor with 2GiB size
    // that fails to allocate in Vulkan (the limit is `2GiB - 31`, determined
    // from `Max Storage Buffer Binding Size` limit determined by wgpu)
    const F32_SIZES: Range<i32> = 7..13;
    #[divan::bench(args = sizes(F32_SIZES))]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left_perm = InputMatrixDimensions::new(1, 2, 0);
        let right_perm = InputMatrixDimensions::new(1, 0, 2);
        let out_perm = Permutation::new(vec![2, 1, 0]);

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = WrappedTensor::<f32>::random(&shape);
        let right = WrappedTensor::<f32>::random(&shape);
        let layer = ConcatMatMul::new_with_permute(left_perm, right_perm, out_perm);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[Shape::from(left.shape()), Shape::from(right.shape())],
                )
                .expect("ConcantMatMul should succeed")
        });
    }
}

#[divan::bench_group]
mod qkv_layer {
    use core::slice;
    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape, Tensor,
        layers::{provable::Evaluate, transformer::qkv::QKV},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes, sizes};

    // XXX: beyond this point benchmarks for elements are too slow, see matmul
    // benches for measurements.
    //
    //                   fastest | slowest | median  | mean    | samples │ iters
    // Args { pow2: 11 } 2.36 m  | 2.36 m  | 2.36 m  | 2.36 m  | 1       | 1
    const ELEMENT_SIZES: Range<i32> = 7..10;

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let num_heads = 1;
        let q = KeyedTensor::new(
            "qkv_weight.q",
            Tensor::<Element>::random(&vec![size, size].into()),
        );
        let q_bias = KeyedTensor::new("qkv_bias.q", Tensor::random(&vec![size].into()));
        let k = KeyedTensor::new("qkv_weight.k", Tensor::random(&vec![size, size].into()));
        let k_bias = KeyedTensor::new("qkv_bias.k", Tensor::random(&vec![size].into()));
        let v = KeyedTensor::new("qkv_weight.v", Tensor::random(&vec![size, size].into()));
        let v_bias = KeyedTensor::new("qkv_bias.v", Tensor::random(&vec![size].into()));

        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        bencher
            .with_inputs(|| {
                // The Element QKV layer has a cache and it works only on first evaluation
                QKV::<Element>::new(
                    q.clone(),
                    Some(q_bias.clone()),
                    k.clone(),
                    Some(k_bias.clone()),
                    v.clone(),
                    Some(v_bias.clone()),
                    num_heads,
                    num_heads,
                )
                .unwrap()
            })
            .bench_refs(|layer| {
                layer
                    .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                    .expect("QKV should succeed");
            });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let num_heads = 1;
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
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = QKV::<f32>::new(
            q,
            Some(q_bias),
            k,
            Some(k_bias),
            v,
            Some(v_bias),
            num_heads,
            num_heads,
        )
        .unwrap();

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("QKV should succeed")
        });
    }
}

#[divan::bench_group]
mod logits_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{provable::Evaluate, transformer::logits::Logits},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let rows = 1 << args.pow2;
        let cols = 16384;
        let shape = Shape::new(vec![rows, cols]);

        let input = WrappedTensor::<Element>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Logits::Argmax;

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Logits should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let rows = 1 << args.pow2;
        let cols = 16384;
        let shape = Shape::new(vec![rows, cols]);

        let input = WrappedTensor::<f32>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Logits::Argmax;
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Logits should succeed")
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

    #[divan::bench(args = highrank())]
    fn element_highrank(bencher: divan::Bencher, args: ArgsHighRank) {
        let shape = Shape::new(vec![args.d0, args.d1, args.d2]);

        let input = WrappedTensor::<Element>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Logits::Argmax;

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Logits high rank should succeed")
        });
    }

    #[divan::bench(args = highrank())]
    fn f32_highrank(bencher: divan::Bencher, args: ArgsHighRank) {
        let shape = Shape::new(vec![args.d0, args.d1, args.d2]);

        let input = WrappedTensor::<f32>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Logits::Argmax;
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Logits high rank should succeed")
        });
    }
}

#[divan::bench_group]
mod norm_layer {
    use core::slice;
    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::layernorm::LayerNorm},
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::DATA_SIZE_POWS;

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

    #[divan::bench(args = args())]
    fn element(bencher: divan::Bencher, args: Args) {
        let dim0 = 1 << args.dim0_pow2;
        let dim1 = 1 << args.dim1_pow2;

        let gamma = KeyedTensor::new(
            "layernom_gamma",
            Tensor::<Element>::random(&Shape::new(vec![dim1])),
        );
        let beta = KeyedTensor::new(
            "layernom_beta",
            Tensor::<Element>::random(&Shape::new(vec![dim1])),
        );
        let layer = LayerNorm::<Element>::new(gamma, beta, EPS);

        let input = Tensor::<Element>::random(&Shape::new(vec![dim0, dim1]));
        let input_shape = slice::from_ref(input.shape());

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let input = WrappedTensor::try_from(&input).unwrap();
        let (layer, _, _) = layer.quantise(input_scaling, input_scaling).unwrap();

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Norm should succeed")
        });
    }

    #[divan::bench(args = args())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let dim0 = 1 << args.dim0_pow2;
        let dim1 = 1 << args.dim1_pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![dim0, dim1]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let gamma = KeyedTensor::new(
            "layernorm_gamma",
            Tensor::<f32>::random(&Shape::new(vec![dim1])),
        );
        let beta = KeyedTensor::new(
            "layernorm_beta",
            Tensor::<f32>::random(&Shape::new(vec![dim1])),
        );
        let layer = LayerNorm::<f32>::new(gamma, beta, EPS);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Norm should succeed")
        });
    }
}

#[divan::bench_group]
mod positional_absolute_layer {
    use core::slice;
    use ff_ext::GoldilocksExt2;
    use zkml::{
        ScalingFactor, Shape, Tensor,
        layers::{
            provable::{Evaluate, QuantizeOp},
            transformer::positional::Positional,
        },
        quantization::AbsoluteMax,
        tensor::{KeyedTensor, WrappedTensor},
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2; // emb = size
        let context = size * 2;

        let pos = KeyedTensor::new(
            "absolute_positional_mat",
            Tensor::<f32>::random(&Shape::new(vec![context, size])),
        );
        let input_f32 = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_scaling = ScalingFactor::from_tensor(&input_f32, None);
        let input = WrappedTensor::try_from(&input_f32.to_quantized(&input_scaling)).unwrap();
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        bencher
            .with_inputs(|| {
                QuantizeOp::quantize_op::<AbsoluteMax>(
                    Positional::<f32>::new_absolute(pos.clone()),
                    &(),
                    0usize.into(),
                    &[input_scaling],
                    &[Shape::new(vec![context, size])],
                )
                .expect("quantize positional absolute should succeed")
                .quantized_op
            })
            .bench_refs(|layer| {
                layer
                    .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                    .expect("Positional absolute should succeed");
            });
    }
    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2; // emb = size
        let context = size * 2;

        let pos = KeyedTensor::new(
            "absolute_positional_mat",
            Tensor::<f32>::random(&Shape::new(vec![context, size])),
        );

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);
        bencher
            .with_inputs(|| Positional::<f32>::new_absolute(pos.clone()))
            .bench_refs(|layer| {
                layer
                    .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                    .expect("Positional absolute should succeed");
            });
    }
}

#[divan::bench_group]
mod positional_rope_layer {
    use core::slice;
    use ff_ext::GoldilocksExt2;
    use std::f32::consts::PI;
    use zkml::{
        ScalingFactor, Shape, Tensor,
        layers::{
            provable::{Evaluate, QuantizeOp},
            transformer::positional::Positional,
        },
        quantization::AbsoluteMax,
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size: usize = 1 << args.pow2;
        let context = size * 2;
        if size < 2 || !size.is_multiple_of(2) {
            return;
        }

        let input_f32 = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_scaling = ScalingFactor::from_tensor(&input_f32, None);
        let input = WrappedTensor::try_from(&input_f32.to_quantized(&input_scaling)).unwrap();
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let num_angles = size / 2;
        let angles: Vec<f32> = (0..num_angles)
            .map(|i| ((i as f32) + 1.0) * (PI / (num_angles as f32 + 1.0)))
            .collect();

        bencher
            .with_inputs(|| {
                QuantizeOp::quantize_op::<AbsoluteMax>(
                    Positional::<f32>::new_rope(
                        angles.clone(),
                        "rope_angles".to_string().into(),
                        context,
                    )
                    .expect("new_rope"),
                    &(),
                    0usize.into(),
                    &[input_scaling],
                    input_shape,
                )
                .expect("quantize positional rope should succeed")
                .quantized_op
            })
            .bench_refs(|layer| {
                layer
                    .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                    .expect("Positional rope should succeed");
            });
    }
    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size: usize = 1 << args.pow2;
        let context = size * 2;
        if size < 2 || !size.is_multiple_of(2) {
            return;
        }

        let num_angles = size / 2;
        let angles: Vec<f32> = (0..num_angles)
            .map(|i| ((i as f32) + 1.0) * (PI / (num_angles as f32 + 1.0)))
            .collect();

        let input = WrappedTensor::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);
        bencher
            .with_inputs(|| {
                Positional::<f32>::new_rope(
                    angles.clone(),
                    "rope_angles".to_string().into(),
                    context,
                )
                .expect("new_rope")
            })
            .bench_refs(|layer| {
                layer
                    .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                    .expect("Positional rope should succeed");
            });
    }
}

#[divan::bench_group]
mod permute_layer {
    use core::slice;
    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{permute::Permute, provable::Evaluate},
        tensor::WrappedTensor,
    };

    use crate::{Args, sizes};

    // XXX: 2**10 fails with `BufferTooBig(8589934592)`
    const SIZES: Range<i32> = 7..10;

    #[divan::bench(args = sizes(SIZES))]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size, size]);

        let input = WrappedTensor::<Element>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Permute::new(vec![2, 1, 0]);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Permute should succeed")
        });
    }

    #[divan::bench(args = sizes(SIZES))]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let shape = Shape::new(vec![size, size, size]);

        let input = WrappedTensor::<f32>::random(&shape);
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Permute::new(vec![2, 1, 0]);
        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Permute should succeed")
        });
    }
}

#[divan::bench_group]
mod softmax_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, ScalingFactor, Shape, Tensor,
        layers::{provable::Evaluate, transformer::softmax::Softmax},
        quantization,
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = Tensor::<Element>::random(&Shape::new(vec![size, size]));
        let input_shape = slice::from_ref(input.shape());

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let input = WrappedTensor::try_from(&input).unwrap();
        let layer = Softmax::<f32>::new(size)
            .quantise(input_scaling, *quantization::BIT_LEN)
            .expect("Softmax quantise should succeed");

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Softmax should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Softmax::new(size);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Softmax should succeed")
        });
    }
}

#[divan::bench_group]
mod requant_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{provable::Evaluate, requant::Requant},
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Requant::from_multiplier(2.0, 8);

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Requant should succeed")
        });
    }
}

#[divan::bench_group]
mod pooling_layer {
    use core::slice;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape,
        layers::{
            pooling::{MAXPOOL2D_KERNEL_SIZE, Maxpool2D, Pooling},
            provable::Evaluate,
        },
        tensor::WrappedTensor,
    };

    use crate::{Args, default_sizes};

    #[divan::bench(args = default_sizes())]
    fn element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<Element>::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Pooling::Maxpool2D(Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        });

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Softmax should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let input = WrappedTensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_shape = Shape::from(input.shape());
        let input_shape = slice::from_ref(&input_shape);

        let layer = Pooling::Maxpool2D(Maxpool2D {
            kernel_size: MAXPOOL2D_KERNEL_SIZE,
            stride: MAXPOOL2D_KERNEL_SIZE,
        });

        bencher.bench(|| {
            layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("Softmax should succeed")
        });
    }
}
#[divan::bench_group]
mod einsum_layer {
    use core::slice;
    use std::ops::Range;

    use ff_ext::GoldilocksExt2;
    use zkml::{
        Element, Shape, Tensor,
        layers::{einsum::EinSum, provable::Evaluate},
        tensor::KeyedTensor,
    };

    use crate::{Args, default_sizes, sizes};

    // XXX: beyond this point benchmarks for elements are too slow, see matmul
    // benches for measurements.
    //
    //                   fastest | slowest | median  | mean    | samples │ iters
    // Args { pow2: 11 } 2.36 m  | 2.36 m  | 2.36 m  | 2.36 m  | 1       | 1
    const ELEMENT_SIZES: Range<i32> = 7..10;
    const CONCATS: usize = 8;

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn einsum_qkv_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        let q = KeyedTensor::new("qkv_weight.q", Tensor::random(&vec![size, size].into()));
        let q_bias = KeyedTensor::new("qkv_bias.q", Tensor::random(&vec![size].into()));
        let k = KeyedTensor::new("qkv_weight.k", Tensor::random(&vec![size, size].into()));
        let k_bias = KeyedTensor::new("qkv_bias.k", Tensor::random(&vec![size].into()));
        let v = KeyedTensor::new("qkv_weight.v", Tensor::random(&vec![size, size].into()));
        let v_bias = KeyedTensor::new("qkv_bias.v", Tensor::random(&vec![size].into()));

        let input = Tensor::<Element>::random(&Shape::new(vec![size, size]));
        let input_shape = slice::from_ref(input.shape());

        let einsum_layer = EinSum::<Element>::new(
            "X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)".to_string(),
            vec![Some(q), Some(k), Some(v)],
            vec![Some(q_bias), Some(k_bias), Some(v_bias)],
        )
        .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("EinSum should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
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

        let input = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input_shape = slice::from_ref(input.shape());

        let einsum_layer = EinSum::<f32>::new(
            "X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)".to_string(),
            vec![Some(q), Some(k), Some(v)],
            vec![Some(q_bias), Some(k_bias), Some(v_bias)],
        )
        .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(&[&input], input_shape)
                .expect("EinSum should succeed")
        });
    }

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn einsum_concat_matmul_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = Tensor::<Element>::random(&shape);
        let right = Tensor::<Element>::random(&shape);
        let einsum_layer =
            EinSum::<Element>::new("A(kij)@B(jil)->C(ikl)".to_string(), vec![None], vec![None])
                .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[left.shape().clone(), right.shape().clone()],
                )
                .expect("EinSum should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn einsum_concat_matmul_f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;

        // concat dim must match the `left_perm` and `right_perm` config
        let shape = Shape::new(vec![size, CONCATS, size]);
        let left = Tensor::<f32>::random(&shape);
        let right = Tensor::<f32>::random(&shape);
        let einsum_layer =
            EinSum::<f32>::new("A(kij)@B(jil)->C(ikl)".to_string(), vec![None], vec![None])
                .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[left.shape().clone(), right.shape().clone()],
                )
                .expect("EinSum should succeed")
        });
    }

    #[divan::bench(args = sizes(ELEMENT_SIZES))]
    fn einsum_matmul_element(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = Tensor::<Element>::random(&vec![size, size].into());
        let right = Tensor::<Element>::random(&vec![size, size].into());
        let bias = KeyedTensor::new("matmul_bias", Tensor::<Element>::random(&vec![size].into()));

        let einsum_layer = EinSum::<Element>::new(
            "A(ij)@B(kj)->C(ik)".to_string(),
            vec![None],
            vec![Some(bias.clone())],
        )
        .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[left.shape().clone(), right.shape().clone()],
                )
                .expect("EinSum should succeed")
        });
    }

    #[divan::bench(args = default_sizes())]
    fn einsum_matmul_f32(bencher: divan::Bencher, args: Args) {
        let size = 1 << args.pow2;
        let left = Tensor::<f32>::random(&vec![size, size].into());
        let right = Tensor::<f32>::random(&vec![size, size].into());
        let bias = KeyedTensor::new("matmul_bias", Tensor::<f32>::random(&vec![size].into()));

        let einsum_layer = EinSum::<f32>::new(
            "A(ij)@B(kj)->C(ik)".to_string(),
            vec![None],
            vec![Some(bias.clone())],
        )
        .unwrap();

        bencher.bench(|| {
            einsum_layer
                .evaluate::<GoldilocksExt2>(
                    &[&left, &right],
                    &[left.shape().clone(), right.shape().clone()],
                )
                .expect("EinSum should succeed")
        });
    }
}

fn main() {
    divan::main();
}
