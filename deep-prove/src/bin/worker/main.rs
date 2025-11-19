use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use deep_prove::{
    middleware::{
        v1::{self, DeepProveRequest as DeepProveRequestV1},
        v2::Provable,
    },
    store::{self, MemStore, S3Store, Store},
};
use std::path::PathBuf;
use tracing::{debug, error, info};
use tracing_subscriber::{EnvFilter, filter::LevelFilter, fmt::format::FmtSpan};
use url::Url;
use zkml::{
    Element, Prover, default_transcript,
    model::Model,
    parser::onnx::FloatOnnxLoader,
    quantization::{AbsoluteMax, ModelMetadata},
};

mod api;
mod immediate;
mod lpn;

/// From a proof request wrapped in a [`DeepProveRequestV1`] and a [`Store`]
/// implementation (to interact with the PPs), attempt to generate proofs for a
/// list of inputs.
async fn run_model_v1<S: Store>(
    model: DeepProveRequestV1,
    mut store: S,
) -> Result<Vec<v1::Output>> {
    info!("Proving inference");
    let DeepProveRequestV1 {
        model,
        model_file_hash,
        input,
        scaling_strategy,
        scaling_input_hash,
    } = model;

    let model_file_hash = model_file_hash.unwrap_or_else(|| {
        let hash = <sha2::Sha256 as sha2::Digest>::digest(&model);
        format!("{hash:X}")
    });

    let params_key = store::ParamsKey {
        model_file_hash: model_file_hash.clone(),
    };
    let model_key = store::ModelKey {
        model_file_hash,
        scaling_strategy,
        scaling_input_hash,
    };

    let params = store
        .get_params(&params_key)
        .await
        .context("fetching PPs")?;
    let is_stored_params = params.is_some();

    let store::ScaledModel {
        model,
        model_metadata,
    } = store
        .clone()
        .get_or_init_model_with(&model_key, async move || {
            let model_bytes = model.clone();
            let (model, model_metadata) =
                tokio::task::spawn_blocking(move || parse_model(model_bytes.as_ref()))
                    .await
                    .context("running parsing model task")?
                    .context("parsing model")?;
            Ok(store::ScaledModel {
                model,
                model_metadata,
            })
        })
        .await
        .context("initializing model")?;

    let inputs = input.to_elements(&model_metadata);

    let (prover_ctx, verifier_ctx, model) = if let Some(store::Params { prover, verifier }) = params
    {
        (prover, verifier, model)
    } else {
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let (prover_ctx, verifier_ctx) =
                model.generate_contexts().context("generating model")?;
            Ok((prover_ctx, verifier_ctx, model))
        })
        .await
        .context("running context generation task")?
        .context("generating context")?
    };

    let (prover_ctx, verifier_ctx) = if !is_stored_params {
        // Since prover_ctx is not `Clone` we store and then retrieve the params
        store
            .insert_params(
                &params_key,
                store::Params {
                    prover: prover_ctx,
                    verifier: verifier_ctx,
                },
            )
            .await
            .context("storing PPs")?;

        let store::Params { prover, verifier } = store
            .get_params(&params_key)
            .await
            .context("fetching PPs after storing")?
            .context("PPs not found after storing")?;

        (prover, verifier)
    } else {
        (prover_ctx, verifier_ctx)
    };

    let proofs = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let mut proofs = vec![];
        for (i, input) in inputs.into_iter().enumerate() {
            debug!("Running input #{i}");
            let input_tensors = model
                .load_input_flat(vec![input])
                .context("loading flat inputs")?;

            let trace_result = model.run(
                input_tensors,
                &mut tenstore::GenStore::new_temporary(1000 * 1024 * 1024)?,
            );
            // If model.run fails, print the error and continue to the next input
            match trace_result {
                Ok(trace) => {
                    let mut prover_transcript = default_transcript();
                    let prover = Prover::<_, _, _>::new(&prover_ctx, &mut prover_transcript);
                    let proof = prover
                        .prove(&trace)
                        .with_context(|| "unable to generate proof for {i}th input")?;
                    let output_handles = trace.outputs();
                    let outputs = output_handles
                        .iter()
                        .map(|handle| handle.tensor().map(|t| t.clone()))
                        .collect::<Result<_, _>>()?;

                    proofs.push(v1::Output {
                        outputs,
                        proof: Provable {
                            proof,
                            io: trace.to_verifier_io().context("generating verifier IOs")?,
                            ctx: verifier_ctx.clone(),
                        },
                    });
                }
                Err(e) => {
                    error!("[!] Error running inference for input {}: {}", i + 1, e);
                    continue; // Skip to the next input without writing to CSV
                }
            };
        }
        Ok(proofs)
    })
    .await
    .context("generating proof")?
    .context("running proof generation task")?;

    info!("Proving done.");
    Ok(proofs)
}

fn parse_model(bytes: &[u8]) -> anyhow::Result<(Model<Element>, ModelMetadata)> {
    let strategy = AbsoluteMax::new();
    FloatOnnxLoader::from_bytes_with_scaling_strategy(bytes, strategy)
        .with_keep_float(true)
        .build()
}

fn setup_logging(json: bool) {
    if json {
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_target(true)
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("Setting up logging failed");
    } else {
        let subscriber = tracing_subscriber::fmt()
            .pretty()
            .compact()
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_target(true)
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .from_env_lossy(),
            )
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("Setting up logging failed");
    };
}

#[derive(Parser)]
#[command(version = deep_prove::get_version!(), about)]
struct Args {
    #[command(subcommand)]
    run_mode: RunMode,
}

#[derive(clap::Args)]
struct S3Args {
    #[arg(long, env, default_value = "ap-northeast-2")]
    s3_region: String,
    #[arg(long, env)]
    s3_params_bucket: Option<String>,
    #[arg(long, env, default_value = "dp-models")]
    s3_models_bucket: String,
    #[arg(long, env, default_value = "http://localhost:9000")]
    s3_endpoint: String,
    #[arg(long, env, default_value = "1000")]
    s3_timeout_secs: u64,
    #[arg(env, default_value = "ma-clef-idiote")]
    s3_access_key_id: String,
    #[arg(env, default_value = "mon-suppose-secret")]
    s3_secret_access_key: String,
    /// Enable local file-system cache for S3 data
    #[arg(long, env, default_value_t = false)]
    fs_cache: bool,
    /// Set the path of the S3 store local cache.
    #[arg(long, env, default_value = "/var/cache")]
    fs_cache_dir: PathBuf,
}

#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum RunMode {
    /// Connect to a LPN gateway to receive inference tasks.
    Http {
        #[arg(long, env, default_value = "http://localhost:4000")]
        gw_url: Url,

        /// Directory to cache downloaded models.
        #[arg(long, env, default_value = "worker_cache/")]
        model_cache_dir: PathBuf,

        /// This worker unique name. If not set, a UID will be tentatively built
        /// from the machine ID.
        #[arg(short, long, env)]
        worker_name: Option<String>,

        /// The operator ETH address.
        #[arg(long, env)]
        address: String,

        /// Print the logs in JSON format.
        #[arg(long, env)]
        json: bool,

        /// If set, use S3 to store & fetch PPs, otherwise use memory.
        #[command(flatten)]
        s3_args: S3Args,
    },
    /// Prove inference on local files
    OneShot {
        /// The model to prove inference on.
        #[arg(short = 'm', long)]
        onnx: PathBuf,

        /// The inputs to prove inference for.
        #[arg(short = 'i', long)]
        inputs: PathBuf,
    },
    /// Run a HTTP server and process requests
    LocalApi {
        /// Listening port
        #[arg(short, long, env, default_value_t = 8080)]
        port: u16,

        /// Print the logs in JSON format
        #[arg(long, env)]
        json: bool,

        /// The maximal proof request size to accept (in MB)
        #[arg(long, env, default_value_t = 200)]
        max_body_size: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.run_mode {
        local_args @ RunMode::OneShot { .. } => immediate::run(local_args).await,
        api_args @ RunMode::LocalApi { .. } => api::serve(api_args).await,
        http_args @ RunMode::Http { .. } => lpn::http::run(http_args).await,
    }
}

#[derive(Clone)]
enum StoreKind {
    S3(S3Store),
    Mem(MemStore),
}
