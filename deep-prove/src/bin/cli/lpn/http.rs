use alloy::{hex, signers::local::PrivateKeySigner};
use alloy_signer::SignerSync;
use anyhow::{Context, bail};
use axum::http::StatusCode;
use deep_prove::middleware::v2::{ClientToGw, TaskClass};
use serde_json::json;
use std::io::Write;
use tracing::{error, info};
use url::Url;
use zeroize::Zeroize;
use zkml::inputs::Input;

use crate::{Command, Executor};

/// Maximum proof size (1GiB)
const MAX_PROOF_SIZE: u64 = 1000 * 1024 * 1024;

pub async fn authenticate(api_url: &Url, private_key: &str) -> anyhow::Result<String> {
    let endpoint = api_url.join("/api/v1/prove/auth")?;
    let signer = PrivateKeySigner::from_slice(
        &hex::decode_to_array::<&str, 32>(private_key).context("parsing private key")?,
    )
    .context("instantiating signer from private key")?;
    let address = signer.address();

    let siwe_msg = ureq::get(endpoint.join(&format!("{endpoint}/{address}"))?.to_string())
        .call()
        .context("calling authentication API, first phase")?
        .body_mut()
        .with_config()
        .limit(MAX_PROOF_SIZE)
        .read_to_string()?;

    let signature = signer
        .sign_message_sync(siwe_msg.as_bytes())
        .context("signing SIWE message")?;

    let token = ureq::post(endpoint.as_str())
        .send_json(json! ({
                "siwe_msg": siwe_msg,
                "siwe_sig": signature.to_string()
        }))
        .context("calling authentication API, second phase")?
        .body_mut()
        .with_config()
        .read_to_string()?;

    Ok(token)
}

pub async fn connect(executor: Executor) -> anyhow::Result<()> {
    let Executor::LpnHttp {
        gw_url,
        mut private_key,
        command,
    } = executor
    else {
        unreachable!()
    };

    let token = authenticate(&gw_url, private_key.expose_secret()).await?;
    private_key.zeroize();

    match command {
        Command::Submit { .. } => bail!("`submit` is not supported"),
        Command::Request {
            pretty_name,
            model_id,
            inputs,
            max_fee,
        } => {
            let input = Input::from_file(&inputs).context("loading input")?;

            let request = ClientToGw {
                pretty_name: pretty_name.unwrap_or_else(|| {
                    format!(
                        "{model_id}-{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::SystemTime::UNIX_EPOCH)
                            .expect("you're not Dr. Who -- come back to the forward-flowing time")
                            .as_secs()
                    )
                }),
                class: TaskClass::RunOnnx {
                    model_id: model_id.try_into().context("`model_id` is too large")?,
                    input,
                },
                max_fee,
            };

            // build the API endpoint request and send the whole thing
            let mut resp = ureq::post(gw_url.join("/api/v1/prove/tasks")?.as_str())
                .header("authorization", &token)
                .send_json(request)
                .context("calling API")?;
            match resp.status() {
                StatusCode::CREATED => {
                    info!("[CREATED] {}", resp.body_mut().read_to_string()?);
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
            let mut resp = ureq::get(gw_url.join("/api/v1/prove/proof")?.as_str())
                .header("authorization", &token)
                .call()
                .context("calling API")?;

            match resp.status() {
                StatusCode::OK => {
                    let filename = filename.unwrap_or_else(|| {
                        format!(
                            "{}.bin",
                            std::time::SystemTime::now()
                                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        )
                    });
                    std::fs::File::create(&filename)
                        .context("failed to create proof file")?
                        .write_all(
                            resp.body_mut()
                                .with_config()
                                .limit(MAX_PROOF_SIZE)
                                .read_to_vec()?
                                .as_slice(),
                        )
                        .context("failed to write proof")?;
                    info!("proof written to {filename}");
                }
                StatusCode::NO_CONTENT => {
                    info!("no proof ready");
                }
                c => {
                    error!(
                        "failed to fetch proof: [{}] {}",
                        c.as_str(),
                        resp.body_mut().read_to_string()?
                    );
                }
            }
        }
        Command::Cancel { task_id } => {
            // build the API endpoint request and send the whole thing
            let mut resp = ureq::delete(
                gw_url
                    .join(format!("/api/v1/proof/tasks/{task_id}").as_str())?
                    .as_str(),
            )
            .header("authorization", &token)
            .call()
            .context("calling API")?;
            match resp.status() {
                StatusCode::NO_CONTENT => {
                    info!("task successfully cancelled");
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
    }

    Ok(())
}
