use crate::{Command, Executor};
use anyhow::{Context, bail};
use std::io::Write;
use tracing::{error, info};
use ureq::http::status::StatusCode;
use zkml::{inputs::Input, quantization::ScalingStrategyKind};

pub async fn connect(executor: Executor) -> anyhow::Result<()> {
    let Executor::LocalApi {
        worker_url,
        command,
    } = executor
    else {
        unreachable!()
    };

    match command {
        Command::Submit { onnx, inputs } => {
            let input = Input::from_file(&inputs).context("loading input")?;
            let model_file = std::fs::File::open(&onnx).context("opening model file")?;
            let model = unsafe { memmap2::Mmap::map(&model_file) }
                .context("mmap-ing model file")?
                .to_vec();
            let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
            let scaling_input_hash = None;

            let request = crate::DeepProveRequestV1 {
                model,
                input,
                scaling_strategy,
                scaling_input_hash,
            };

            // build the API endpoint and send the whole thing
            let mut resp = ureq::post(worker_url.join("/proofs")?.as_str())
                .send_json(request)
                .context("sending proof request to the worker")?;
            match resp.status() {
                StatusCode::CREATED => {
                    info!("{}", resp.body_mut().read_to_string()?);
                }
                c => {
                    error!(
                        "failed to send request: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
        Command::Fetch { filename } => {
            const DEFAULT_PREFIX: &str = "proof-";
            // Build the endpoint URL
            let mut resp = ureq::get(worker_url.join("/proofs")?.as_str()).call()?;

            match resp.status() {
                StatusCode::OK => {
                    // create a file to write the proofs to
                    let mut file = tempfile::Builder::new()
                        .prefix(filename.as_deref().unwrap_or(DEFAULT_PREFIX))
                        .suffix(".json")
                        .rand_bytes(10)
                        .disable_cleanup(true)
                        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;

                    // save the list of proofs
                    let body = resp
                        .body_mut()
                        .with_config()
                        .limit(200 * 1024 * 1024)
                        .read_to_vec()?;
                    file.write_all(&body)?;
                    info!("proof received, saved to {}", file.path().display());
                }
                StatusCode::NO_CONTENT => {
                    info!("no proof ready yet");
                }
                c => {
                    // these status codes should never be produced by the worker
                    error!("unknown status: {}", c.as_str())
                }
            }
        }
        Command::Request { .. } => bail!("`request` is not supported"),
        Command::Cancel { .. } => bail!("`cancel` is not supported"),
    }
    Ok(())
}
