use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use deep_prove::{
    middleware::{
        v1::{self, DeepProveRequest as DeepProveRequestV1, T},
        v2::Provable,
    },
    store::{self, MemStore, S3Store, Store},
};
use std::{net::SocketAddr, path::PathBuf};
use tenstore::GenStore;
use tracing::{Span, debug, info};
use url::Url;
use zkml::{
    Element, Prover,
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
    mut model_data_store: S,
    mut tenstore: GenStore,
    proof_id: String,
) -> Result<Vec<v1::Output>> {
    info!("Proving inference");
    let DeepProveRequestV1 {
        model,
        model_file_hash,
        input,
        scaling_strategy,
        scaling_input_hash,
    } = model;

    info!(
        "Received proof job: inputs={} scaling_strategy={:?}",
        input.len(),
        scaling_strategy
    );
    let model_file_hash = model_file_hash.unwrap_or_else(|| {
        let hash = <sha2::Sha256 as sha2::Digest>::digest(&model);
        format!("{hash:X}")
    });
    debug!("Computed model hash: {model_file_hash}");

    let params_key = store::ParamsKey {
        model_file_hash: model_file_hash.clone(),
    };
    let model_key = store::ModelKey {
        model_file_hash,
        scaling_strategy,
        scaling_input_hash,
    };

    let params = model_data_store
        .get_params(&params_key)
        .await
        .context("fetching PPs")?;
    let is_stored_params = params.is_some();
    let store::ScaledModel {
        model,
        model_metadata,
    } = model_data_store
        .clone()
        .get_or_init_model_with(&model_key, || async move {
            let parse_model_span = Span::current();
            let model_bytes = model.clone();
            debug!("Parsing model bytes and preparing metadata");
            let (model, model_metadata) = tokio::task::spawn_blocking(move || {
                let _enter = parse_model_span.enter();
                parse_model(model_bytes.as_ref())
            })
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
    let layer_count = model.graph().inner_nodes().count();
    let input_count = model.num_inputs();
    let output_count = model.graph().output_nodes().count();
    info!(
        "Model prepared: inputs={} outputs={} layers={}",
        input_count, output_count, layer_count
    );

    let inputs = input.to_elements(&model_metadata);

    let (prover_ctx, verifier_ctx, model) = if let Some(store::Params { prover, verifier }) = params
    {
        info!("Using stored proving and verifier contexts");
        (prover, verifier, model)
    } else {
        let ctx_span = Span::current();
        info!("Generating proving and verifier contexts");
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let _enter = ctx_span.enter();
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
        model_data_store
            .insert_params(
                &params_key,
                store::Params {
                    prover: prover_ctx,
                    verifier: verifier_ctx,
                },
            )
            .await
            .context("storing PPs")?;
        info!("Stored generated proving parameters for reuse");

        let store::Params { prover, verifier } = model_data_store
            .get_params(&params_key)
            .await
            .context("fetching PPs after storing")?
            .context("PPs not found after storing")?;

        (prover, verifier)
    } else {
        (prover_ctx, verifier_ctx)
    };

    let parent_span = Span::current();
    let proofs = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let _parent_guard = parent_span.enter();
        let mut proofs = vec![];
        for (i, input) in inputs.into_iter().enumerate() {
            debug!(input_index = i, proof_id = %proof_id, "Running input");
            let input_tensors = model
                .load_input_flat(vec![input])
                .context("loading flat inputs")?;

            let trace = model
                .run(input_tensors, &mut tenstore)
                .context(format!("Running inference for input {}", i + 1))?;
            let output_handles = trace.outputs();
            let outputs = output_handles
                .iter()
                .map(|handle| handle.tensor().map(|t| t.clone()))
                .collect::<Result<_, _>>()?;
            let io = trace.to_verifier_io().context("generating verifier IOs")?;
            let proof = Prover::<_, T, _>::prove(&prover_ctx, trace, &model)
                .with_context(|| "unable to generate proof for {i}th input")?;

            proofs.push(v1::Output {
                outputs,
                proof: Provable {
                    proof,
                    io,
                    ctx: verifier_ctx.clone(),
                },
            });
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

#[derive(Parser)]
#[command(version = deep_prove::get_version!(), about)]
struct Args {
    #[command(subcommand)]
    run_mode: RunMode,

    /// Tensor store kind. One of: temporary, local, remote
    #[arg(long, value_enum, required = true)]
    tensor_store: TenStoreKind,

    /// Tensor store in-memory cache size in bytes. Defaults to 1 MiB
    #[arg(long, default_value = "1048576")]
    store_cache_size: usize,

    /// Tensor store file-system cache root dir
    #[arg(long)]
    store_root_dir: Option<PathBuf>,

    /// Tensor remote store server address.
    #[arg(long)]
    store_server_addr: Option<SocketAddr>,
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

        /// The maximum size of a job response from the gateway.
        #[arg(long, env, default_value_t = 100 * 1024 * 1024)] // 100MB
        max_job_size: u64,

        /// If set, use S3 to store & fetch PPs, otherwise use memory.
        #[command(flatten)]
        s3_args: S3Args,
    },
    /// Prove inference on local files
    #[command()]
    OneShot {
        /// The model to prove inference on.
        #[arg(short = 'm', long = "model", required = true)]
        model: PathBuf,

        /// Format of the supplied model
        /// currently supported: onnx, gguf, safetensors
        #[arg(long, value_enum, required = true)]
        model_format: ModelFormat,

        /// The inputs to prove inference for (only valid for ONNX).
        #[arg(
            short = 'i',
            long,
            required_if_eq("model_format", "onnx"),
            conflicts_with_all = ["prompt", "tokenizer", "config", "max_new_tokens"]
        )]
        inputs: Option<PathBuf>,

        /// Prompt to prove for LLM models (gguf/safetensors).
        #[arg(
            long,
            required_if_eq_any([("model_format", "gguf"), ("model_format", "safetensors")]),
            conflicts_with = "inputs"
        )]
        prompt: Option<String>,

        /// Path to tokenizer.json (required for safetensors).
        #[arg(
            long,
            required_if_eq("model_format", "safetensors"),
            requires = "config",
            conflicts_with = "inputs"
        )]
        tokenizer: Option<PathBuf>,

        /// Path to config.json (required for safetensors).
        #[arg(
            long,
            required_if_eq("model_format", "safetensors"),
            requires = "tokenizer",
            conflicts_with = "inputs"
        )]
        config: Option<PathBuf>,

        /// Maximum number of tokens to generate (LLM only).
        #[arg(
            long,
            default_value_t = 8,
            conflicts_with = "inputs",
            requires = "prompt"
        )]
        max_new_tokens: usize,
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
    let Args {
        run_mode,
        tensor_store,
        store_cache_size,
        store_root_dir,
        store_server_addr,
    } = Args::parse();

    let tenstore = match tensor_store {
        TenStoreKind::Temporary => GenStore::new_temporary(store_cache_size),
        TenStoreKind::Local => GenStore::new_local(
            store_root_dir.context("Must specify cache dir for local store")?,
            store_cache_size,
        ),
        TenStoreKind::Remote => GenStore::new_remote(
            store_root_dir.context("Must specify cache dir for local store")?,
            store_cache_size,
            store_server_addr.context("Must server address for remote store")?,
        ),
    }?;

    match run_mode {
        local_args @ RunMode::OneShot { .. } => immediate::run(local_args, tenstore).await,
        api_args @ RunMode::LocalApi { .. } => api::serve(api_args, tenstore).await,
        http_args @ RunMode::Http { .. } => lpn::http::run(http_args, tenstore).await,
    }
}

#[derive(Clone)]
enum StoreKind {
    S3(S3Store),
    Mem(MemStore),
}

#[derive(Copy, Clone, clap::ValueEnum)]
pub enum ModelFormat {
    Onnx,
    Gguf,
    Safetensors,
}

/// Tensor store kind
#[derive(Copy, Clone, clap::ValueEnum)]
enum TenStoreKind {
    Temporary,
    Local,
    Remote,
}
