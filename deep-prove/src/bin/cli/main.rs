use anyhow::Context;
use clap::{Parser, Subcommand};
use deep_prove::middleware::v1::{DeepProveRequest as DeepProveRequestV1, Output};
use redact::Secret;
use std::{io::BufReader, path::PathBuf};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;
use url::Url;

mod local;
mod lpn;

#[derive(Parser)]
#[command(version = deep_prove::get_version!(), about)]
struct Args {
    #[command(subcommand)]
    executor: Executor,
}

#[derive(Subcommand)]
enum Executor {
    /// Interact with a LPN gateway with the HTTP.
    LpnHttp {
        /// The URL of the LPN gateway.
        #[arg(short, long, env)]
        gw_url: Url,

        /// The client ETH private key.
        #[clap(short, long, env)]
        private_key: Secret<String>,

        #[command(subcommand)]
        command: Command,
    },

    /// Interact with the API exposed by a prover.
    LocalApi {
        /// The root URL of the worker
        #[arg(short, long, env, default_value = "http://localhost:8080")]
        worker_url: Url,

        #[command(subcommand)]
        command: Command,
    },

    /// Verify that a proof is correct.
    Verify {
        /// The file containing the serialized proof to verify.
        #[arg(long, short)]
        proof: PathBuf,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Submit a model and its input to prove inference.
    Submit {
        /// Path to the ONNX file of the model to prove.
        #[arg(short = 'm', long)]
        onnx: PathBuf,

        /// Path to the inputs  to prove inference for.
        #[arg(short, long)]
        inputs: PathBuf,
    },

    /// Submit inputs to be proved for an existing model.
    Request {
        /// The user-facing name of this request. Will default to a timestamp if not set.
        #[arg(short = 'p', long = "pretty")]
        pretty_name: Option<String>,

        /// The ID of the model to prove the inference for.
        #[arg(short, long)]
        model_id: usize,

        /// Path to the inputs to prove inference for.
        #[arg(short, long)]
        inputs: PathBuf,

        /// The maximal price to pay (in $LA) for the task to be executed.
        #[arg(long)]
        max_fee: u128,
    },

    /// If it has not yet been processed, cancel this task.
    Cancel {
        /// The UUID of the task to cancel.
        task_id: uuid::Uuid,
    },

    /// Fetch a generated proof, if any are available.
    Fetch {
        /// The file to write the proof to - if empty, use the proof ID.
        filename: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .pretty()
        .compact()
        .with_level(true)
        .with_file(false)
        .with_line_number(false)
        .with_target(false)
        .without_time()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber).context("Setting up logging failed")?;

    let args = Args::parse();

    match args.executor {
        http_config @ Executor::LpnHttp { .. } => lpn::http::connect(http_config).await,
        local_config @ Executor::LocalApi { .. } => local::connect(local_config).await,
        Executor::Verify { proof } => {
            info!("verifying proof in `{}`", proof.display());
            let output: Vec<Output> = serde_json::from_reader(BufReader::new(
                std::fs::File::open(&proof)
                    .with_context(|| format!("failed to open `{}`", proof.display()))?,
            ))
            .context("deserializing proof")?;

            output.into_iter().try_fold((), |_, o| o.proof.verify())
        }
    }
}
