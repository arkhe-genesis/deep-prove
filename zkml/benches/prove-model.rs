use criterion::{Criterion, criterion_group, criterion_main};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use tenstore::GenStore;
use zkml::{
    Element, Prover, ProverContext, Tensor,
    inputs::Input,
    iop::context::VerifierContext,
    model::{
        Model,
        llm::{Driver, LLMTokenizerObserver},
    },
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::{
            LLMTokenizer,
            models::gpt2::{GPT2, GPT2_Q8_0},
            tokenizer::TokenizerLoader,
        },
        onnx::FloatOnnxLoader,
    },
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

type P<'a, 'b> = Prover<'a, 'b, F, Transcript, Pcs<F>>;

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

fn random_input(inputs: &[Vec<Tensor<Element>>]) -> Vec<Tensor<Element>> {
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
    let (model, inputs) = parse_model_and_inputs(MLP_IRIS, inputs);
    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs<F>>()
        .expect("unable to generate context");

    group.bench_with_input("mlp", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || random_input(inputs),
            |inputs| prove_model(model, inputs, &prover_ctx, &verifier_ctx),
            criterion::BatchSize::SmallInput,
        )
    });

    let inputs = zstd::Decoder::new(CNN_CIFAR_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(CNN_CIFAR, inputs);
    let (prover_ctx, verifier_ctx) = model
        .generate_contexts::<F, Pcs<F>>()
        .expect("unable to generate context");

    group.bench_with_input("cnn", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || random_input(inputs),
            |inputs| prove_model(model, inputs, &prover_ctx, &verifier_ctx),
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
            || random_input(inputs),
            |inputs| model.run::<F>(inputs, &mut GenStore::default()).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    let inputs = zstd::Decoder::new(CNN_CIFAR_INPUT).expect("failed to parse zstd");
    let (model, inputs) = parse_model_and_inputs(CNN_CIFAR, inputs);

    group.bench_with_input("cnn", &(model, inputs), |bencher, (model, inputs)| {
        bencher.iter_batched(
            || random_input(inputs),
            |inputs| model.run::<F>(inputs, &mut GenStore::default()).unwrap(),
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

    // Setting the max context to `2` so that only a single run is performed.
    let max_context = 2;

    let model_path = file_cache::from_cache(GPT2_Q8_0).expect("failed to find GPT2 model in cache");
    let format = RawGGUF::new(model_path);
    let driver = Driver::load_from_model(GPT2, &format, Some(max_context))
        .expect("failed to instantiate GPT2 driver");
    let tokenizer = GPT2.load_tokenizer(&format).unwrap();
    let user_tokens = driver.random_sequence(1);
    let sentence = tokenizer.detokenize(&user_tokens);

    group.bench_function("gpt2", |bencher| {
        bencher.iter_with_large_drop(|| {
            driver.run::<GoldilocksExt2>(
                &user_tokens,
                &mut GenStore::default(),
                Some(LLMTokenizerObserver {
                    input: sentence.to_string(),
                    tokenizer: &tokenizer,
                }),
            )
        });
    });

    group.finish();
}

// NOTE: XXX: when running, limit RAYON_NUM_THREADS to e.g. 2 to avoid high
// concurrency resulting in measure noise.
criterion_group!(benches, inference, prove);
criterion_main!(benches);
