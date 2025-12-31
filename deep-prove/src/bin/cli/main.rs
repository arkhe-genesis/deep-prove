use anyhow::Context;
use bincode::serde::decode_from_slice;
use clap::{Parser, Subcommand, ValueEnum};
use deep_prove::middleware::{
    llm::LlmOneShotOutput,
    v1::{DeepProveRequest as DeepProveRequestV1, Output},
};
use redact::Secret;
use std::{fs, path::PathBuf};
use tracing::{info, info_span};
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
    /// Authenticate to a LPN gateway and store the authentication token.
    Authenticate {
        /// The URL of the LPN gateway.
        #[arg(short, long, env, default_value = "http://localhost:4000")]
        gw_url: Url,

        /// The client ETH private key.
        #[clap(short, long, env)]
        private_key: Secret<String>,

        /// Where to store the authentication token.
        #[clap(short, long, env, default_value = "lpn-token.txt")]
        token_path: PathBuf,
    },
    /// Interact with a LPN gateway with the HTTP.
    LpnHttp {
        /// The URL of the LPN gateway.
        #[arg(short, long, env, default_value = "http://localhost:4000")]
        gw_url: Url,

        /// How to authenticate to the LPN gateway.
        #[clap(flatten)]
        auth: AuthMethod,

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

#[derive(clap::Args)]
#[group(required = true, multiple = false)]
pub struct AuthMethod {
    /// Authenticate to the LPN gateway with the provided token.
    #[clap(short, long, env)]
    pub token_path: Option<PathBuf>,
    /// Authenticate to the LPN gateway with this private key.
    #[clap(short, long, env)]
    pub private_key: Option<Secret<String>>,
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
    let _telemetry_guard = telemetry::setup_logging("deep-prove-cli", false);
    let args = Args::parse();

    match args.executor {
        Executor::Authenticate {
            gw_url,
            private_key,
            token_path,
        } => lpn::http::save_token(&gw_url, private_key.expose_secret(), &token_path).await,
        http_config @ Executor::LpnHttp { .. } => {
            let _span = info_span!("dp_cli_lpn_http").entered();
            lpn::http::connect(http_config).await
        }
        local_config @ Executor::LocalApi { .. } => {
            let _span = info_span!("dp_cli_local_api").entered();
            local::connect(local_config).await
        }
        Executor::Verify { proof } => {
            let _span = info_span!("dp_cli_verify", proof_path = %proof.display()).entered();
            info!("verifying proof in `{}`", proof.display());
            verify_proof(proof)
        }
    }
}

#[derive(Copy, Clone, ValueEnum)]
pub enum ProofFormat {
    Json,
    Bin,
}

fn verify_proof(proof: PathBuf) -> anyhow::Result<()> {
    let bytes = fs::read(&proof)
        .with_context(|| format!("failed to read proof file `{}`", proof.display()))?;

    let format = match proof
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ext) if ext == "json" => ProofFormat::Json,
        _ => ProofFormat::Bin,
    };

    match format {
        ProofFormat::Json => {
            let outputs: Vec<Output> =
                serde_json::from_slice(&bytes).context("deserializing ONNX proof (JSON)")?;
            outputs.into_iter().try_fold((), |_, o| o.proof.verify())
        }
        ProofFormat::Bin => {
            let (llm, _) =
                decode_from_slice::<LlmOneShotOutput, _>(&bytes, bincode::config::standard())
                    .context("deserializing LLM proof (bincode)")?;

            info!(
                "verifying LLM proof for model {} with prompt {:?}",
                llm.model_name, llm.prompt
            );
            if let Some(text) = &llm.llm_response {
                info!("LLM response: {}", text);
            }

            llm.verifier
                .verify(llm.proof.proof, llm.tokens, llm.proof.io)
        }
    }
}
