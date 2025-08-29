use core::slice;
use std::ops::{Range, RangeInclusive};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use zkml::{
    Element, ScalingFactor, Shape, Tensor,
    layers::{
        activation::GELU,
        add::Add,
        convolution::Convolution,
        dense::Dense,
        flatten::Flatten,
        matrix_mul::{self, MatMul},
        provable::Evaluate,
        transformer::{embeddings::Embeddings, qkv::QKV},
    },
    tensor::Number,
};

const DATA_SIZE_POWS: Range<i32> = 7..14;

fn add_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = Tensor::<Element>::random(&shape);
        let input = Tensor::<Element>::random(&shape);

        let layer = Add::<Element>::new_with(operand.clone(), shape.clone());

        group.bench_with_input(
            BenchmarkId::new("add/Element", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Add should succeed")
                });
            },
        );
    }

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let shape = Shape::new(vec![size, size]);
        let operand = Tensor::<f32>::random(&shape);
        let input = Tensor::<f32>::random(&shape);

        let layer = Add::<f32>::new_with(operand.clone(), shape.clone());

        group.bench_with_input(
            BenchmarkId::new("add/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Add should succeed")
                });
            },
        );
    }

    group.finish();
}

fn dense_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let matrix = Tensor::<Element>::random(&Shape::new(vec![size, size]));
        let bias = Tensor::<Element>::random(&Shape::new(vec![size]));
        let input = Tensor::<Element>::random(&Shape::new(vec![size]));

        let dense = Dense::<Element>::new(matrix.clone(), bias.clone());

        group.bench_with_input(
            BenchmarkId::new("dense/Element", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    dense
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Dense should succeed")
                });
            },
        );
    }

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let matrix = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let bias = Tensor::<f32>::random(&Shape::new(vec![size]));
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));

        let dense = Dense::<f32>::new(matrix.clone(), bias.clone());

        group.bench_with_input(
            BenchmarkId::new("dense/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    dense
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Dense should succeed")
                });
            },
        );
    }
}

fn convolution_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    let batches = 1;
    let channels = 3;
    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let input = Tensor::<f32>::random(&Shape::new(vec![batches, channels, size, size]));
        let kernels = Tensor::<f32>::random(&Shape::new(vec![batches, channels, 3, 3]));
        let bias = Tensor::<f32>::random(&Shape::new(vec![batches]));

        let input_shape = input.shape();
        let convolution = Convolution::<f32>::new(kernels.clone(), bias.clone());

        group.bench_with_input(
            BenchmarkId::new("convolution/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    convolution
                        .evaluate::<GoldilocksExt2>(&[input], slice::from_ref(&input_shape))
                        .expect("Convolution should succeed")
                });
            },
        );
    }

    // NOTE: as it is currently implemented, conv2d_i performs one kernel invocation per output.
    // the maximum supported input size on a M2 is 2**12.
    let range = 7..12;
    let batches = 1;
    let channels = 3;
    for pow2 in range {
        let size = 1 << pow2;
        let input = Tensor::<Element>::random(&Shape::new(vec![batches, channels, size, size]));
        let kernels = Tensor::<Element>::random(&Shape::new(vec![batches, channels, 3, 3]));
        let bias = Tensor::<Element>::random(&Shape::new(vec![batches]));

        let input_shape = input.shape();
        let convolution = Convolution::<Element>::new(kernels.clone(), bias.clone())
            .into_padded_and_ffted(&input_shape);

        group.bench_with_input(
            BenchmarkId::new("convolution/Element", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    convolution
                        .evaluate::<GoldilocksExt2>(&[input], slice::from_ref(&input_shape))
                        .expect("Convolution should succeed")
                });
            },
        );
    }

    group.finish();
}

fn embeddings_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for pow2 in DATA_SIZE_POWS {
        let vocab_size = 100;
        let size = 1 << pow2;
        let emb = Tensor::<Element>::random(&Shape::new(vec![vocab_size, size]));
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));
        let scaling = ScalingFactor::from_span(
            <f32 as Number>::MIN,
            <f32 as Number>::MAX,
            Some((0, vocab_size as Element)),
        );
        let input = input.quantize(&scaling);

        let layer = Embeddings::<Element>::new(emb.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("embeddings/Element", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Embeddings should succeed")
                });
            },
        );
    }

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let emb = Tensor::<f32>::random(&Shape::new(vec![size, size]));
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));

        let layer = Embeddings::<f32>::new(emb.clone()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("embeddings/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[])
                        .expect("Embeddings should succeed")
                });
            },
        );
    }

    group.finish();
}

fn flatten_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    const DATA_SIZE_POWS: Range<i32> = 4..7;
    const RANKS: RangeInclusive<usize> = 2..=4;

    for pow2 in DATA_SIZE_POWS {
        for rank in RANKS {
            let size = 1 << pow2;
            let input = Tensor::<Element>::random(&Shape::new([size].repeat(rank)));
            let layer = Flatten;

            group.bench_function(
                BenchmarkId::new("flatten/Element", format!("{size}^{rank}")),
                |b| {
                    b.iter(|| {
                        layer
                            .evaluate::<GoldilocksExt2>(&[&input], &[])
                            .expect("Flatten should succeed")
                    });
                },
            );
        }
    }

    for pow2 in DATA_SIZE_POWS {
        for rank in RANKS {
            let size = 1 << pow2;
            let input = Tensor::<f32>::random(&Shape::new([size].repeat(rank)));
            let layer = Flatten;

            group.bench_function(
                BenchmarkId::new("flatten/f32", format!("{size}^{rank}")),
                |b| {
                    b.iter(|| {
                        layer
                            .evaluate::<GoldilocksExt2>(&[&input], &[])
                            .expect("Flatten should succeed")
                    });
                },
            );
        }
    }
}

fn gelu_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for size in [1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 22] {
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));
        let gelu = GELU::<f32>::new();

        group.bench_with_input(BenchmarkId::new("gelu", size), &input, |b, input| {
            b.iter(|| {
                gelu.evaluate::<GoldilocksExt2>(&[input], &[])
                    .expect("GeLU should succeed")
            });
        });
    }
}

fn matrix_mul_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let left = Tensor::<Element>::random(&vec![size, size].into());
        let right = Tensor::<Element>::random(&vec![size, size].into());
        let bias = Tensor::<Element>::random(&vec![size].into());
        let config = matrix_mul::Config::TransposeB;

        let layer = MatMul::<Element>::new_with_config(
            matrix_mul::OperandMatrix::Input,
            matrix_mul::OperandMatrix::Input,
            Some(bias),
            config,
        )
        .unwrap();

        group.bench_function(
            BenchmarkId::new("matrix_mul/Element", format!("{size}x{size}")),
            |b| {
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[&left, &right], &[]));
            },
        );
    }

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;
        let left = Tensor::<f32>::random(&vec![size, size].into());
        let right = Tensor::<f32>::random(&vec![size, size].into());
        let bias = Tensor::<f32>::random(&vec![size].into());
        let config = matrix_mul::Config::TransposeB;

        let layer = MatMul::<f32>::new_with_config(
            matrix_mul::OperandMatrix::Input,
            matrix_mul::OperandMatrix::Input,
            Some(bias),
            config,
        )
        .unwrap();

        group.bench_function(
            BenchmarkId::new("matrix_mul/f32", format!("{size}x{size}")),
            |b| {
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[&left, &right], &[]));
            },
        );
    }
}

fn qkv_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("run-layers");
    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;

        let num_heads = 1;
        let q = Tensor::<Element>::random(&vec![1, size].into());
        let q_bias = Tensor::random(&vec![size].into());
        let k = Tensor::random(&vec![1, size].into());
        let k_bias = Tensor::random(&vec![size].into());
        let v = Tensor::random(&vec![1, size].into());
        let v_bias = Tensor::random(&vec![size].into());
        let input = Tensor::<Element>::random(&Shape::new(vec![size, size]));

        let layer = QKV::<Element>::new(q, q_bias, k, k_bias, v, v_bias, num_heads).unwrap();

        group.bench_with_input(
            BenchmarkId::new("qkv/Element", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[Shape::new(vec![size, size])])
                        .expect("QKV should succeed")
                });
            },
        );
    }

    for pow2 in DATA_SIZE_POWS {
        let size = 1 << pow2;

        let num_heads = 1;
        let q = Tensor::<f32>::random(&vec![1, size].into());
        let q_bias = Tensor::random(&vec![size].into());
        let k = Tensor::random(&vec![1, size].into());
        let k_bias = Tensor::random(&vec![size].into());
        let v = Tensor::random(&vec![1, size].into());
        let v_bias = Tensor::random(&vec![size].into());
        let input = Tensor::<f32>::random(&Shape::new(vec![size, size]));

        let layer = QKV::<f32>::new(q, q_bias, k, k_bias, v, v_bias, num_heads).unwrap();

        group.bench_with_input(
            BenchmarkId::new("qkv/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| {
                    layer
                        .evaluate::<GoldilocksExt2>(&[input], &[Shape::new(vec![size, size])])
                        .expect("QKV should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    add_layer,
    convolution_layer,
    dense_layer,
    embeddings_layer,
    flatten_layer,
    gelu_layer,
    matrix_mul_layer,
    qkv_layer,
);
criterion_main!(benches);
