use super::{WorkerResources, instantiate_store, proving::process_job};
use crate::RunMode;
use anyhow::{Context, anyhow, ensure};
use base64::{Engine, prelude::BASE64_STANDARD};
use deep_prove::middleware::v2;
use exponential_backoff::Backoff;
use serde_json::json;
use tenstore::GenStore;
use tracing::{Instrument, debug, error, info, info_span, warn};
use url::Url;

const ATTEMPTS: u32 = 5;
const MIN_WAIT_MS: u64 = 1000;
const MAX_WAIT_MS: u64 = 100000;

/// How long to wait when a GW-linked error occured before polling again.
const IDLE_POLL_INTERVAL_MS: u64 = 5000;

pub fn retry_operation<F, T, E: std::fmt::Debug>(func: F, log: impl Fn() -> String) -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
{
    for duration in Backoff::new(
        ATTEMPTS,
        std::time::Duration::from_millis(MIN_WAIT_MS),
        std::time::Duration::from_millis(MAX_WAIT_MS),
    ) {
        let result = func();
        match &result {
            Ok(_) => {
                return result;
            }
            Err(e) => match duration {
                Some(duration) => {
                    warn!(
                        "failed to execute operation. operation: {} retry_secs: {} err: {:?}",
                        log(),
                        duration.as_secs(),
                        &e
                    );
                    std::thread::sleep(duration);
                }
                None => {
                    error!("eventually failed to execute operation {}", log());
                    return result;
                }
            },
        }
    }

    unreachable!()
}

/// A minimal struct to extract just the job_id from a potentially malformed job.
#[derive(serde::Deserialize)]
struct PartialJob {
    job_id: i64,
}

/// A wrapper for the connection settings, as well as helper functions to
/// interact with a gateway.
struct ConnContext {
    gw_url: Url,
    worker_name: String,
    address: String,
    max_job_size: u64,
}

/// Job response data from the gateway.
struct JobResponse {
    body: String,
    traceparent: Option<String>,
}

impl ConnContext {
    fn new(gw_url: Url, worker_name: String, address: String, max_job_size: u64) -> Self {
        let address = address.trim_start_matches("0x").to_string();
        Self {
            gw_url,
            worker_name,
            address,
            max_job_size,
        }
    }

    /// Request a new job from the gateway, returning the raw response body
    /// and any trace headers attached by the gateway.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn request_job(&self) -> anyhow::Result<JobResponse> {
        let response = ureq::get(
            self.gw_url
                .join(&format!("api/v1/jobs/{}", self.worker_name))
                .unwrap()
                .as_str(),
        )
        .header("authorization", &self.address)
        .call()
        .context("connecting to gateway")?;

        let traceparent = response
            .headers()
            .get("traceparent")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        let body = response
            .into_body()
            .into_with_config()
            .limit(self.max_job_size)
            .read_to_string()
            .context("reading response body")?;

        Ok(JobResponse { body, traceparent })
    }

    /// Confirm to the GW that we successfully received the job.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn ack_job(&self, job_id: i64) -> anyhow::Result<()> {
        retry_operation(
            || {
                telemetry::ureq_inject_trace_headers(ureq::get(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/ack", self.worker_name))
                        .unwrap()
                        .as_str(),
                ))
                .header("authorization", &self.address)
                .call()
            },
            || format!("ACK-ing job #{job_id}"),
        )?;

        Ok(())
    }

    /// Submit the proof for the given `job_id`.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn submit_proof(&self, job_id: i64, proof: &[u8]) -> anyhow::Result<()> {
        let encoded_proof = BASE64_STANDARD.encode(proof);
        info!(
            "submitting a {} proof",
            humansize::format_size(encoded_proof.len(), humansize::DECIMAL)
        );
        retry_operation(
            || {
                telemetry::ureq_inject_trace_headers(ureq::put(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/proof", self.worker_name))
                        .unwrap()
                        .as_str(),
                ))
                .header("authorization", &self.address)
                .send_json(json!({
                    "proof": BASE64_STANDARD.encode(proof),
                }))
            },
            || format!("sending proof for job #{job_id} to the gateway"),
        )?;

        Ok(())
    }

    /// Submit a failure message for the given `job_id`.
    ///
    ///  - Will fail if the connection settings are not valid.
    ///  - Will fail after retries if the connection can not be established.
    fn submit_error(&self, job_id: i64, err_msg: &str) -> anyhow::Result<()> {
        retry_operation(
            || {
                telemetry::ureq_inject_trace_headers(ureq::put(
                    self.gw_url
                        .join(&format!("/api/v1/jobs/{}/{job_id}/error", self.worker_name))
                        .unwrap()
                        .as_str(),
                ))
                .header("authorization", &self.address)
                .send_json(json!({
                    "error": err_msg,
                }))
            },
            || format!("sending error for job #{job_id} to the gateway"),
        )?;

        Ok(())
    }
}

pub async fn run(args: crate::RunMode, tenstore: GenStore) -> anyhow::Result<()> {
    let RunMode::Http {
        gw_url,
        address,
        json,
        worker_name,
        model_cache_dir,
        max_job_size,
        s3_args,
    } = args
    else {
        unreachable!()
    };
    let _telemetry_guard = telemetry::setup_logging("deep-prove-worker", json);

    let worker_name = worker_name
        .ok_or(anyhow!("no worker name set"))
        .or_else(|_| machine_uid::get())
        .map_err(|_| anyhow!("failed to generate a unique worker name"))?;
    ensure!(
        !worker_name.is_empty(),
        "failed to generate a non-empty worker name"
    );
    info!("gateway URL: {gw_url}");
    info!("operator address: {address}");
    info!("worker unique name: {worker_name}");
    info!(
        "max job size: {}",
        humansize::format_size(max_job_size, humansize::BINARY)
    );

    let WorkerResources { model_fetcher } = instantiate_store(&s3_args, model_cache_dir.clone())
        .context("initializing worker resources")?;
    let conn = ConnContext::new(gw_url, worker_name, address, max_job_size);

    loop {
        let job_tenstore = tenstore.start_new_run();
        // 1. Request job from the GW
        debug!("waiting for task from gateway");
        let response = match conn.request_job() {
            Ok(body) => body,
            Err(e) => {
                error!("failed to fetch job from gateway: {e:?}");
                std::thread::sleep(std::time::Duration::from_millis(IDLE_POLL_INTERVAL_MS));
                continue;
            }
        };
        let JobResponse { body, traceparent } = response;

        // 2. Parse the job, handling deserialization errors gracefully
        let job = match serde_json::from_str::<v2::GwToWorker>(&body) {
            Ok(job) => job,
            Err(e) => {
                error!("failed to deserialize job from gateway: {e}");
                // Try to extract just the job_id to report the error back
                if let Ok(partial) = serde_json::from_str::<PartialJob>(&body) {
                    let err_msg = format!("worker failed to deserialize job: {e}");
                    if let Err(submit_err) = conn.submit_error(partial.job_id, &err_msg) {
                        error!("failed to submit error to gateway: {submit_err:?}");
                    } else {
                        info!(
                            "submitted deserialization error for job #{}",
                            partial.job_id
                        );
                    }
                } else {
                    error!("could not extract job_id from malformed response to report error");
                }
                std::thread::sleep(std::time::Duration::from_millis(IDLE_POLL_INTERVAL_MS));
                continue;
            }
        };

        let job_id = job.job_id;
        let job_span = info_span!("gw_job", proof_id = job_id);
        let trace_headers: Vec<_> = traceparent
            .iter()
            .map(|v| ("traceparent".to_string(), v.clone()))
            .collect();
        telemetry::set_parent_from_headers(&job_span, &trace_headers);

        async {
            info!("received job #{job_id} to execute");

            // 3. ACK job
            match conn.ack_job(job_id) {
                Ok(_) => debug!("ACK-ed job #{job_id}"),
                Err(err) => error!("failed to ACK job: {err:?}"),
            }

            // 4. Process job & submit proof
            match process_job(job, job_tenstore.clone(), &model_fetcher).await {
                Ok(proof) => {
                    if proof.is_empty() {
                        let err_msg = format!("proof payload empty for job {job_id}");
                        conn.submit_error(job_id, &err_msg)
                            .context("submitting error to gateway")?;
                        info!("submitted error for job #{job_id}");
                    } else {
                        conn.submit_proof(job_id, &proof)
                            .context("submitting proofs to gateway")?;
                        info!("submitted proof for job #{job_id}");
                    }
                }
                Err(err) => {
                    conn.submit_error(job_id, &format!("{err:?}"))
                        .context("submitting error to gateway")?;
                    error!("submitted error: {err:?} for job #{job_id}");
                }
            }
            Ok::<_, anyhow::Error>(())
        }
        .instrument(job_span)
        .await?;
    }
}
