//! This module implements a prover instance that spins up an HTTP API, then
//! listen for proof request from there. Proofs are generated locally, and
//! made available for collection through an API endpoint.
//! All the state is in-memory, there is no persistence layer.

//! API endpoints are:
//! - GET  /status       returns the current count of waiting and proved tasks;
//! - GET  /proofs       returns the first proof from the ready proofs queue, if any;
//! - POST /proofs       receive a JSON-serialized proof request wrapped in a [`DeepProveRequestV1`];
//! - GET  /healthcheck  returns a 200 OK response, as long as the process is alive.
use std::sync::Arc;

use crate::RunMode;
use anyhow::Context;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
};
use base64::{Engine, prelude::BASE64_STANDARD};
use deep_prove::{middleware::v1::DeepProveRequest as DeepProveRequestV1, store::MemStore};
use tenstore::GenStore;
use tokio::sync::Mutex;
use tracing::{error, info, info_span, trace};

#[derive(Default)]
struct AppState {
    work_queue: Vec<DeepProveRequestV1>,
    proofs_queue: Vec<String>,
}

/// Expose a long-lived local HTTP API, executing submitted proofs and returning
/// ready proofs on request.
pub async fn serve(args: RunMode, tenstore: GenStore) -> anyhow::Result<()> {
    let RunMode::LocalApi {
        port,
        json,
        max_body_size,
    } = args
    else {
        unreachable!()
    };
    let _telemetry_guard = telemetry::setup_logging("deep-prove-worker", json);

    #[cfg(feature = "aws-marketplace")]
    {
        use std::{env, time};

        let config = aws_config::load_from_env().await;
        let client = aws_sdk_marketplacemetering::Client::new(&config);
        let aws_product_code =
            env::var("AWS_PRODUCT_CODE").context("getting AWS marketplace product code")?;
        let aws_pk_version: i32 = env::var("AWS_PK_VERSION")
            .context("getting AWS marketplace public key version")?
            .parse()
            .context("Parsing `AWS_PK_VERSION` into i32")?;
        let nonce = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();
        // This will trigger metering when ran within AWS ECS or EKS
        client
            .register_usage()
            .product_code(aws_product_code)
            .public_key_version(aws_pk_version)
            .nonce(nonce)
            .send()
            .await
            .context("AWS marketplace error registering usage: {err}")?;
    }

    let app_state = Arc::new(Mutex::new(AppState::default()));

    {
        let app_state = app_state.clone();
        tokio::spawn(async move {
            let store = MemStore::default();
            loop {
                let maybe_work = { app_state.lock().await.work_queue.pop() };
                if let Some(proof_request) = maybe_work {
                    let request_span =
                        info_span!("dp_worker_prove_inference", proof_id = "api-local");
                    let _entered = request_span.enter();
                    let now = std::time::Instant::now();
                    info!("processing proof...");
                    let job_tenstore = tenstore.start_new_run();
                    let result = crate::run_model_v1(
                        proof_request,
                        store.clone(),
                        job_tenstore.clone(),
                        "api-local".to_string(),
                    )
                    .await;
                    match result {
                        Ok(proofs) => {
                            info!("proof generated in {}s", now.elapsed().as_secs());
                            app_state
                                .lock()
                                .await
                                .proofs_queue
                                .extend(proofs.into_iter().map(|proof| {
                                    BASE64_STANDARD.encode(
                                        postcard::to_allocvec(&proof)
                                            .context("serializing proof")
                                            .unwrap(),
                                    )
                                }));
                        }
                        Err(err) => error!("failed to generate proof: {err:?}"),
                    }
                } else {
                    trace!("no proof request");
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
    }

    let app = Router::new()
        .route(
            "/status",
            get(|State(state): State<Arc<Mutex<AppState>>>| async move {
                let state = state.lock().await;
                (
                    StatusCode::OK,
                    format!(
                        "tasks in queue: {}\nproofs ready: {}",
                        state.work_queue.len(),
                        state.proofs_queue.len()
                    ),
                )
            }),
        )
        .route(
            "/proofs",
            get(|State(state): State<Arc<Mutex<AppState>>>| async move {
                let mut state = state.lock().await;
                if let Some(proof) = state.proofs_queue.pop() {
                    info!("returning a {}MB proof", proof.len() / (1024 * 1024));
                    (StatusCode::OK, proof)
                } else {
                    info!("no proofs ready");
                    (StatusCode::NO_CONTENT, "no proof ready".to_string())
                }
            }),
        )
        .route(
            "/proofs",
            post(
                |State(state): State<Arc<Mutex<AppState>>>,
                 Json(proof_request): Json<DeepProveRequestV1>| async move {
                    let mut state = state.lock().await;
                    info!("adding proof request to the queue");
                    state.work_queue.push(proof_request);
                    (StatusCode::CREATED, "proof submitted")
                },
            )
            .layer(DefaultBodyLimit::max(max_body_size * 1024 * 1024)),
        )
        .route(
            "/healthcheck",
            get(|| async move { (StatusCode::OK, "OK") }),
        )
        .with_state(app_state.clone());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}"))
        .await
        .with_context(|| format!("listening on port {port}"))?;
    axum::serve(listener, app)
        .await
        .context("setting up HTTP server")?;

    Ok(())
}
