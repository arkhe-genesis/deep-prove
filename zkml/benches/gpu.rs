use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use zkml::{
    Element, Tensor,
    layers::{activation::GELU, add::Add, dense::Dense, provable::Evaluate},
    tensor::Shape,
};

fn add_layer(c: &mut Criterion) {
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

    for pow2 in 7..14 {
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

    for pow2 in 7..14 {
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

    for size in [1 << 5, 1 << 10, 1 << 15, 1 << 20, 1 << 22] {
        let input = Tensor::<f32>::random(&Shape::new(vec![size]));
        let gelu = GELU::<f32>::new();

        group.bench_with_input(BenchmarkId::new("gelu", size), &input, |b, input| {
            b.iter(|| gelu.evaluate::<GoldilocksExt2>(&[input], &[]));
        });
    }

    for pow2 in 7..14 {
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

    for pow2 in 7..14 {
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

    group.finish();
}

criterion_group!(benches, add_layer, dense_layer);
criterion_main!(benches);
