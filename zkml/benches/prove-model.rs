use criterion::{Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use tenstore::GenStore;
use zkml::{
    Element, Prover, Tensor, default_transcript,
    inputs::Input,
    model::{Model, Trace},
    parser::onnx::FloatOnnxLoader,
    quantization::{AbsoluteMax, ModelMetadata},
    verify,
};

type F = GoldilocksExt2;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
type Pcs<E> = Basefold<E, BasefoldRSParams>;

type Transcript = transcript::basic::BasicTranscript<F>;

const MLP_IRIS: &[u8] = include_bytes!("../assets/scripts/MLP/mlp-iris-01.onnx");
const MLP_IRIS_INPUT: &[u8] = include_bytes!("../assets/scripts/MLP/input.json.zst");
const CNN_CIFAR: &[u8] = include_bytes!("../assets/scripts/CNN/cnn-cifar-01.onnx");
const CNN_CIFAR_INPUT: &[u8] = include_bytes!("../assets/scripts/CNN/input.json.zst");
// const CNN_COVID: &[u8] = include_bytes!("../assets/scripts/covid/cnn-covid.onnx");
// const CNN_COVID_INPUT: &[u8] = include_bytes!("../assets/scripts/covid/input.json.zst");

fn new_transcript() -> Transcript {
    default_transcript()
}

fn parse_model(model_data: &[u8]) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    FloatOnnxLoader::from_bytes_with_scaling_strategy(model_data, AbsoluteMax::new())
        .with_keep_float(true)
        .build()
}

fn parse_model_and_inputs<T: std::io::Read>(
    model_data: &[u8],
    inputs: T,
) -> (Model<Element>, Vec<Vec<Tensor<Element>>>) {
    let run_inputs = Input::from_reader(inputs).expect("failed to load inputs");
    let (model, md) = parse_model(model_data).expect("failed to parse model");
    let inputs = run_inputs
        .to_elements(&md)
        .into_iter()
        .map(|input| {
            model
                .load_input_flat(vec![input])
                .expect("failed to call load_input_flat on the model")
        })
        .collect();
    (model, inputs)
}

fn prove_model(model: &Model<Element>, inputs: Vec<Vec<Tensor<i64>>>) {
    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs<F>>()
        .expect("unable to generate context");

    for (i, inputs) in inputs.into_iter().enumerate() {
        let trace = model
            .run(inputs, &mut GenStore::default())
            .unwrap_or_else(|_| panic!("input #{i} failed"));

        let mut prover_transcript = new_transcript();
        let prover = Prover::<_, _, _>::new(&prover_ctx, &mut prover_transcript);
        let io = trace.to_verifier_io().unwrap();
        let proof = prover.prove(&trace).expect("unable to generate proof");

        let mut verifier_transcript = new_transcript();
        verify(&verifier_ctx, proof, io, &mut verifier_transcript).expect("invalid proof");
    }
}

fn infer_model(model: &Model<Element>, inputs: Vec<Vec<Tensor<i64>>>) {
    for (i, inputs) in inputs.into_iter().enumerate() {
        let _trace: Trace<F, Element, Element> = model
            .run(inputs, &mut GenStore::default())
            .unwrap_or_else(|_| panic!("input #{i} failed"));
    }
}

fn prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove");

    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    let inputs = zstd::Decoder::new(MLP_IRIS_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(MLP_IRIS, inputs);

    group.bench_with_input("mlp", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || inputs.clone(),
            |inputs| prove_model(model, inputs),
            criterion::BatchSize::SmallInput,
        )
    });

    let inputs = zstd::Decoder::new(CNN_CIFAR_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(CNN_CIFAR, inputs);

    group.bench_with_input("cnn", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || inputs.clone(),
            |inputs| prove_model(model, inputs),
            criterion::BatchSize::SmallInput,
        )
    });

    // NOTE: disabling covid model, as it is /very/ memory-intensive
    // let inputs = zstd::Decoder::new(CNN_COVID_INPUT).expect("failed to parse zstd");
    // let (model, inputs) = parse_model_and_inputs(CNN_COVID, inputs);
    // group.bench_with_input("covid", &(model, inputs), |bencher, (model, inputs)| {
    //     bencher.iter_batched(
    //         || inputs.clone(),
    //         |inputs| infer_model(model, inputs),
    //         criterion::BatchSize::SmallInput,
    //     )
    // });

    group.finish();
}

fn inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    let inputs = zstd::Decoder::new(MLP_IRIS_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(MLP_IRIS, inputs);

    group.bench_with_input("mlp", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || inputs.clone(),
            |inputs| infer_model(model, inputs),
            criterion::BatchSize::SmallInput,
        )
    });

    let inputs = zstd::Decoder::new(CNN_CIFAR_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(CNN_CIFAR, inputs);

    group.bench_with_input("cnn", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || inputs.clone(),
            |inputs| infer_model(model, inputs),
            criterion::BatchSize::SmallInput,
        )
    });

    // NOTE: model parsing fails
    // let inputs = zstd::Decoder::new(CNN_COVID_INPUT).expect("failed to parse zstd");
    // let (model, inputs) = parse_model_and_inputs(CNN_COVID, inputs);
    // group.bench_with_input("covid", &(model, inputs), |bencher, (model, inputs)| {
    //     bencher.iter_batched(
    //         || inputs.clone(),
    //         |inputs| infer_model(model, inputs),
    //         criterion::BatchSize::SmallInput,
    //     )
    // });

    group.finish();
}

// NOTE: XXX: when running, limit RAYON_NUM_THREADS to e.g. 2 to avoid high
// concurrency resulting in measure noise.
criterion_group!(benches, inference, prove);
criterion_main!(benches);
