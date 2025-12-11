use criterion::{Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use tenstore::GenStore;
use zkml::{
    Element, Prover, ProverContext, Tensor,
    inputs::Input,
    iop::context::VerifierContext,
    model::{Model, llm::Driver},
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::models::gpt2::{GPT2, GPT2_Q8_0},
        onnx::FloatOnnxLoader,
    },
    quantization::{AbsoluteMax, ModelMetadata},
    verify,
};

type F = GoldilocksExt2;
type Pcs<E> = Basefold<E, BasefoldRSParams>;

type Transcript = transcript::basic::BasicTranscript<F>;

const MLP_IRIS: &[u8] = include_bytes!("../assets/scripts/MLP/mlp-iris-01.onnx");
const MLP_IRIS_INPUT: &[u8] = include_bytes!("../assets/scripts/MLP/input.json.zst");
const CNN_CIFAR: &[u8] = include_bytes!("../assets/scripts/CNN/cnn-cifar-01.onnx");
const CNN_CIFAR_INPUT: &[u8] = include_bytes!("../assets/scripts/CNN/input.json.zst");

type P<'a, 'b> = Prover<'a, 'b, F, Transcript, Pcs<F>>;

fn parse_model_and_inputs<T: std::io::Read>(
    model_data: &[u8],
    inputs: T,
) -> (Model<Element>, ModelMetadata, Input) {
    let run_inputs = Input::from_reader(inputs).expect("failed to load inputs");
    let (model, metadata) =
        FloatOnnxLoader::from_bytes_with_scaling_strategy(model_data, AbsoluteMax::new())
            .with_keep_float(true)
            .build()
            .expect("failed to parse model");
    (model, metadata, run_inputs)
}

fn inputs_to_elements(
    model: &Model<Element>,
    metadata: &ModelMetadata,
    run_inputs: Input,
) -> Vec<Vec<Tensor<Element>>> {
    run_inputs
        .to_elements(metadata)
        .into_iter()
        .map(|input| {
            model
                .load_input_flat(vec![input])
                .expect("failed to call load_input_flat on the model")
        })
        .collect()
}

fn random_input<T: Clone>(inputs: &[T]) -> T {
    let el = rand::random_range(0..inputs.len());
    inputs[el].clone()
}

fn prove_model(
    model: &Model<Element>,
    inputs: Vec<Tensor<i64>>,
    prover_ctx: &ProverContext<F, Pcs<F>>,
    verifier_ctx: &VerifierContext<F, Pcs<F>>,
) {
    let trace = model.run(inputs, &mut GenStore::default()).unwrap();
    let io = trace.to_verifier_io().unwrap();

    let proof = P::prove(prover_ctx, trace, model).expect("unable to generate proof");

    verify::<_, Transcript, _>(verifier_ctx, proof, io).expect("invalid proof");
}

fn prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove");

    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(80));

    let inputs = zstd::Decoder::new(MLP_IRIS_INPUT).expect("failed to parse zstd");
    let (model, metadata, inputs) = parse_model_and_inputs(MLP_IRIS, inputs);
    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs<F>>()
        .expect("unable to generate context");

    let inputs = inputs_to_elements(&model, &metadata, inputs);
    group.bench_with_input("mlp", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || random_input(inputs),
            |inputs| prove_model(model, inputs, &prover_ctx, &verifier_ctx),
            criterion::BatchSize::SmallInput,
        )
    });

    let inputs = zstd::Decoder::new(CNN_CIFAR_INPUT).expect("failed to parse zstd");
    let (model, _metadata, inputs) = parse_model_and_inputs(CNN_CIFAR, inputs);
    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs<F>>()
        .expect("unable to generate context");

    let inputs = inputs_to_elements(&model, &metadata, inputs);
    group.bench_with_input("cnn", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || random_input(inputs),
            |inputs| prove_model(model, inputs, &prover_ctx, &verifier_ctx),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    group
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(200));

    for (name, max_context) in [("gpt2", 2), ("gpt2_small_run", 10)] {
        let model_path =
            file_cache::from_cache(GPT2_Q8_0).expect("failed to find GPT2 model in cache");
        let format = RawGGUF::new(model_path);
        let driver = Driver::load_from_model(GPT2, &format, Some(max_context))
            .expect("failed to instantiate GPT2 driver");
        let user_tokens = driver.random_sequence(1);

        let driver_f32: &Driver<f32> = &driver;
        let inputs_f32 = vec![driver_f32.tokens_to_tensor(&user_tokens).unwrap()];
        group.bench_function(format!("{name}/f32"), |bencher| {
            bencher.iter_batched(
                || inputs_f32.clone(),
                |inputs| driver_f32.run(inputs, &mut GenStore::default()),
                criterion::BatchSize::SmallInput,
            );
        });

        let (driver_elt, _metadata) = driver
            .into_provable_llm(None)
            .expect("Driver should be provable");
        let driver_elt: Driver<Element> = driver_elt;
        let inputs_elt = vec![driver_elt.tokens_to_tensor(&user_tokens).unwrap()];
        group.bench_function(format!("{name}/Element"), |bencher| {
            bencher.iter_batched(
                || inputs_elt.clone(),
                |inputs| driver_elt.run(inputs, &mut GenStore::default()),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// NOTE: XXX: when running, limit RAYON_NUM_THREADS to e.g. 2 to avoid high
// concurrency resulting in measure noise.
criterion_group!(benches, inference, prove);
criterion_main!(benches);
