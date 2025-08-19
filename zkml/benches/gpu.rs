use std::ops::Range;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use zkml::{
    Element, ScalingFactor, Tensor,
    layers::{
        activation::GELU, add::Add, convolution::Convolution, dense::Dense, provable::Evaluate,
        transformer::embeddings::Embeddings,
    },
    tensor::{Number, Shape},
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
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[input], &[]));
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
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[input], &[]));
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
                b.iter(|| dense.evaluate::<GoldilocksExt2>(&[input], &[]));
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
                b.iter(|| dense.evaluate::<GoldilocksExt2>(&[input], &[]));
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

        let dense = Convolution::<f32>::new(kernels.clone(), bias.clone());

        group.bench_with_input(
            BenchmarkId::new("convolution/f32", format!("{size}x{size}")),
            &input,
            |b, input| {
                b.iter(|| dense.evaluate::<GoldilocksExt2>(&[input], &[]));
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
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[input], &[]));
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
                b.iter(|| layer.evaluate::<GoldilocksExt2>(&[input], &[]));
            },
        );
    }

    group.finish();
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
            b.iter(|| gelu.evaluate::<GoldilocksExt2>(&[input], &[]));
        });
    }
}

criterion_group!(
    benches,
    add_layer,
    dense_layer,
    convolution_layer,
    embeddings_layer,
    gelu_layer
);
criterion_main!(benches);
