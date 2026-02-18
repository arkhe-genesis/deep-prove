use crate::LocalStore;
use anyhow::{Context, Result, anyhow};
use base64::{prelude::BASE64_STANDARD, read::DecoderReader, write::EncoderWriter};
use exponential_backoff::Backoff;
use reqwest::StatusCode;
use serde_json::json;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
    thread::{self, JoinHandle},
    time::Duration,
};
use telemetry::reqwest_inject_trace_headers;
use tokio::{
    select,
    sync::{mpsc, oneshot},
    time::timeout,
};
use tracing::{debug, error, info, warn};
use url::Url;
use urlencoding::encode;
use uuid::Uuid;

use crate::StoreError;
use sha2::{Digest, Sha256};

fn hashed_tensor_key<K: Display>(key: &K) -> String {
    let digest = Sha256::digest(key.to_string().as_bytes());
    format!("{digest:x}")
}

pub async fn retry_async_operation<F, Fut, T, E: std::fmt::Debug>(
    func: F,
    log: impl Fn() -> String,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    const ATTEMPTS: u32 = 5;
    const MIN_WAIT_MS: u64 = 1000;
    const MAX_WAIT_MS: u64 = 100000;

    for duration in Backoff::new(
        ATTEMPTS,
        std::time::Duration::from_millis(MIN_WAIT_MS),
        std::time::Duration::from_millis(MAX_WAIT_MS),
    ) {
        let result = func().await;
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

#[derive(Debug)]
pub struct RemoteClient<K> {
    cmd_tx: mpsc::UnboundedSender<Cmd<K>>,
    worker_handle: Option<JoinHandle<()>>,
    /// Storage keys that are currently being sent to server
    sending: HashSet<K>,
}

#[derive(Debug)]
struct ClientWorker<L, K> {
    local_store: L,
    server_addr: Url,
    cmd_rx: mpsc::UnboundedReceiver<Cmd<K>>,
    /// Storage keys that are currently being prefetched. A call to get the same
    /// key will attach a subscriber to it.
    prefetching: HashMap<K, Option<GetSubTx>>,
    prefetch_sub_rx: mpsc::UnboundedReceiver<Prefetched<K>>,
    prefetch_sub_tx: mpsc::UnboundedSender<Prefetched<K>>,
    monitor: TaskMonitor,
}

type TaskMonitor = super::metrics::TaskMonitor<anyhow::Result<()>>;

#[derive(Debug)]
struct Prefetched<K> {
    storage_key: K,
    result: Result<Vec<u8>>,
}

#[derive(Debug)]
enum Cmd<K> {
    Kill,
    Get {
        run_id: Uuid,
        storage_key: K,
        res_tx: oneshot::Sender<Result<Vec<u8>>>,
    },
    Put {
        run_id: Uuid,
        storage_key: K,
        data: Vec<u8>,
        res_tx: oneshot::Sender<Result<()>>,
    },
    Prefetch {
        run_id: Uuid,
        storage_key: K,
    },
    CleanUp {
        run_id: Uuid,
    },
}

/// Sender to a `GET` command subscriber from an in-progress task
type GetSubTx = oneshot::Sender<Result<Vec<u8>>>;

impl<K> RemoteClient<K>
where
    K: 'static + Clone + Hash + Eq + Display + Send + Sync,
{
    pub fn new<L>(server_addr: Url, local_store: L) -> Result<Self>
    where
        L: 'static + LocalStore + Send,
        L: LocalStore<Key = K>,
    {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (prefetch_sub_tx, prefetch_sub_rx) = mpsc::unbounded_channel();

        let mut worker = ClientWorker {
            local_store,
            server_addr,
            cmd_rx,
            prefetching: HashMap::new(),
            prefetch_sub_rx,
            prefetch_sub_tx,
            monitor: super::metrics::TaskMonitor::new(),
        };
        let async_rt = tokio::runtime::Runtime::new().context("Start client async runtime")?;
        let worker_handle = thread::spawn(move || async_rt.block_on(worker.run()));

        Ok(RemoteClient {
            cmd_tx,
            worker_handle: Some(worker_handle),
            sending: HashSet::new(),
        })
    }

    /// Request data from the server without waiting for the result.
    ///
    /// This is a no-op if the given `storage_key` is already being received and it's still in progress.
    pub fn prefetch(&mut self, run_id: Uuid, storage_key: K) -> Result<()> {
        debug!("Prefetching {storage_key}");
        self.cmd_tx
            .send(Cmd::Prefetch {
                run_id,
                storage_key,
            })
            .context("Sending PREFETCH cmd to worker")?;
        Ok(())
    }

    /// Request data from the server.
    ///
    /// If this `storage_key` is already being prefetched it will internally subscribe to get its result instead.
    ///
    /// Invariant: This be called with the same `storage_key` simultaneously, otherwise the second call will fail.
    pub fn get(&mut self, run_id: Uuid, storage_key: K) -> Result<Vec<u8>> {
        debug!("Getting {storage_key}");
        let (res_tx, res_rx) = oneshot::channel();

        self.cmd_tx
            .send(Cmd::Get {
                run_id,
                storage_key,
                res_tx,
            })
            .context("Sending GET cmd to worker")?;
        res_rx
            .blocking_recv()
            .context("Receiving GET cmd result from worker")?
    }

    /// Put data on the server.
    ///
    /// This is a no-op if the given `storage_key` is already being sent and it's still in progress.
    pub fn put(&mut self, run_id: Uuid, storage_key: K, data: Vec<u8>) -> Result<()> {
        debug!("Putting {storage_key}");
        if self.sending.contains(&storage_key) {
            debug!("Another call to put {storage_key} is already in-progress");
            // Another task sending this is already in-progress
            return Ok(());
        }

        self.sending.insert(storage_key.clone());

        // NOTE: Do not use try operator. We need to avoid early return to correctly update `self.sending` before returning.
        let res = {
            let (res_tx, res_rx) = oneshot::channel();
            self.cmd_tx
                .send(Cmd::Put {
                    run_id,
                    storage_key: storage_key.clone(),
                    data,
                    res_tx,
                })
                .context("Sending PUT cmd to worker")
                .and_then(|()| {
                    res_rx
                        .blocking_recv()
                        .context("Receiving PUT cmd result from worker")
                })
                .flatten()
        };

        self.sending.remove(&storage_key);

        res
    }

    /// Request to clean-up the given run's data from server
    pub fn clean_up(&self, run_id: Uuid) -> Result<()> {
        self.cmd_tx
            .send(Cmd::CleanUp { run_id })
            .context("Sending clean-up cmd to worker")
    }
}

impl<L, K> ClientWorker<L, K>
where
    L: LocalStore<Key = K>,
    K: 'static + Clone + Hash + Eq + Display + Send + Sync,
{
    async fn run(&mut self) {
        loop {
            select! {
                // Handle finished prefetch.
                //
                // NOTE: We don't need to handle None` case as this could also
                // happen during clean-up as the sender (`prefetch_sub_tx`) is
                // also owned by Self. When that happens this branch will be
                // disabled and the other branch will break the loop when its
                // sender (owned by `RemoteClient`) is also dropped.
                Some(prefetched) = self.prefetch_sub_rx.recv() => {
                    let Prefetched { storage_key, result } = prefetched;
                    // Send the result to a subscriber if any attached
                    if let Some(sub) = self.prefetching.remove(&storage_key).expect("Prefetching in-progress must have a map entry") {
                        // If the rx is closed there's nothing left to do
                        let _ = sub.send(result);
                    } else if let Ok(data) = result && let Err(err) = self.local_store.store(storage_key, data) {
                        error!("Failed to store prefetched data locally: {err}")
                    }
                },
                // Handle cmds from the client
                cmd = self.cmd_rx.recv() => {
                    match cmd {
                        None | Some(Cmd::Kill) => {
                            // Make sure that all requests have been processed, especially the clean up.
                            self.monitor.join_all().await;
                            // Exit if the client's channel is closed
                            break
                        },
                        Some(Cmd::Get {
                            run_id,
                            storage_key,
                            res_tx,
                        }) => {
                            self
                                .get(run_id, &storage_key, res_tx)
                                .await;

                        }
                        Some(Cmd::Put {
                            run_id,
                            storage_key,
                            data,
                            res_tx,
                        }) => {
                             self
                                .put(run_id, &storage_key, data, res_tx).await;
                        }
                        Some(Cmd::Prefetch {run_id, storage_key }) => {
                            self
                                .prefetch(run_id, storage_key.clone())
                                .await;
                        }
                        Some(Cmd::CleanUp {run_id}) => {
                            self.clean_up(run_id).await;
                        }
                    }
                }
                // Handle monitored task
                task = Self::join_next_task(&mut self.monitor) => {
                    if let Some((result, task_name)) = task {
                        match result {
                            Ok(Ok(())) => {},
                            Ok(Err(err)) => {
                                error!("Task {task_name} failed with {err}");
                            },
                            Err(err) => {
                                if err.is_cancelled() {
                                    info!("Task cancelled: {task_name}");
                                } else {
                                    debug_assert!(err.is_panic());
                                    std::panic::resume_unwind(err.into_panic());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Join next async task. Returns `None` if there are no tasks
    async fn join_next_task(
        monitor: &mut TaskMonitor,
    ) -> Option<(
        Result<anyhow::Result<()>, tokio::task::JoinError>,
        Cow<'static, str>,
    )> {
        // Timeout to avoid dead-locks from tasks spawning more tasks
        if let Ok(result) = timeout(Duration::from_millis(50), monitor.join_next()).await {
            return result;
        }
        None
    }

    async fn prefetch(&mut self, run_id: Uuid, key: K) {
        if self.local_store.contains(&key) || self.prefetching.contains_key(&key) {
            return;
        }

        self.prefetching.insert(key.clone(), None);

        let server_addr = self.server_addr.clone();
        let prefetch_sub_tx = self.prefetch_sub_tx.clone();
        self.monitor
            .spawn(format!("prefetch/{run_id}/{key}").into(), async move {
                let result = Self::fetch(server_addr, run_id, key.clone()).await;

                let prefetched = Prefetched {
                    storage_key: key,
                    result,
                };
                // If the rx is closed there's nothing left to do
                let _ = prefetch_sub_tx.send(prefetched);
                Ok(())
            })
            .expect("to spawn prefetch task");
    }

    async fn clean_up(&mut self, run_id: Uuid) {
        let _ = self.local_store.clean_up(run_id);
        let server_addr = self.server_addr.clone();
        // A task to wait for transfer from server
        self.monitor
            .spawn(format!("clean-up/{run_id}").into(), async move {
                retry_async_operation(
                    move || {
                        reqwest_inject_trace_headers(
                            reqwest::Client::new().post(
                                server_addr
                                    .join(&format!("/tenstore/{run_id}"))
                                    .unwrap()
                                    .as_str(),
                            ),
                        )
                        .send()
                    },
                    || format!("failed to clean-up remote store for {run_id}"),
                )
                .await
                .context("calling clean-up for remote store")?
                .error_for_status()
                .map(|_| ())
                .context("cleaning up remote store")
            })
            .expect("to spawn clean-up task");
    }

    async fn get(&mut self, run_id: Uuid, key: &K, res_tx: oneshot::Sender<Result<Vec<u8>>>) {
        // Try fetch from local, fallback to remote if not found
        if let Ok(data) = self.local_store.fetch(key.clone()).cloned() {
            res_tx.send(Ok(data)).expect("Sending GET result to client");
            return;
        }

        // Check if the key is being prefetched
        if let Some(sub) = self.prefetching.get_mut(key) {
            if sub.is_some() {
                res_tx
                    .send(Err(anyhow!(
                        "The storage_key {key} is already being received from another call"
                    )))
                    .expect("Sending GET result to client");
                return;
            }

            // This storage is being prefetched. Subscribe to receive its result.
            let (sub_tx, sub_rx) = oneshot::channel();
            *sub = Some(sub_tx);

            // A task to wait for prefetch to finish
            self.monitor
                .spawn(
                    format!("get-wait-prefetch/{run_id}/{key}").into(),
                    async move {
                        let res = sub_rx.await.context("Receiving prefetched data").flatten();
                        // Send back the result to the client
                        res_tx.send(res).expect("Sending GET result to client");
                        Ok(())
                    },
                )
                .expect("to spawn get waiting for prefetch task");
            return;
        }

        let server_addr = self.server_addr.clone();
        let key = key.clone();
        // A task to wait for transfer from server
        self.monitor
            .spawn(format!("get/{run_id}/{key}").into(), async move {
                let result = Self::fetch(server_addr, run_id, key.clone()).await;

                // Send back the result to the client
                res_tx.send(result).expect("Sending GET result to client");
                Ok(())
            })
            .expect("to spawn get task");
    }

    async fn fetch(server_addr: Url, run_id: Uuid, key: K) -> Result<Vec<u8>> {
        let _ = tracing::info_span!(
            "tenstore_fetch",
            run_id = run_id.to_string(),
            key = key.to_string()
        );
        let path_key = hashed_tensor_key(&key);
        let fetch_url = server_addr.join(&format!("/tenstore/{run_id}/{}", encode(&path_key)))?;

        let response = retry_async_operation(
            || reqwest_inject_trace_headers(reqwest::Client::new().get(fetch_url.as_str())).send(),
            || format!("fetching {run_id}/{key}"),
        )
        .await
        .with_context(|| format!("calling fetch for {run_id}/{key}"))?;

        if response.status() == StatusCode::NOT_FOUND {
            return Err(anyhow!(StoreError::RemoteKeyNotFound));
        }
        if let Err(err) = response.error_for_status_ref() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read body".to_string());
            error!("fetching {run_id}/{key}: HTTP {status} - {body}");
            return Err(err)
                .with_context(|| format!("fetching {run_id}/{key}: HTTP {status} - {body}"));
        }
        let response = response
            .error_for_status()
            .with_context(|| format!("fetching {run_id}/{key}"))?;

        let bytes = response.bytes().await?;
        zstd::stream::decode_all(DecoderReader::new(bytes.as_ref(), &BASE64_STANDARD))
            .with_context(|| format!("decoding {run_id}/{key} ({} bytes)", bytes.len()))
    }

    async fn put(
        &mut self,
        run_id: Uuid,
        key: &K,
        data: Vec<u8>,
        res_tx: oneshot::Sender<Result<()>>,
    ) {
        let _ = tracing::info_span!(
            "tenstore_put",
            run_id = run_id.to_string(),
            key = key.to_string()
        );
        let path_key = hashed_tensor_key(key);
        let put_url = self
            .server_addr
            .join(&format!("/tenstore/{run_id}/{}", encode(&path_key)))
            .unwrap();

        let mut encoder = EncoderWriter::new(Vec::with_capacity(data.len()), &BASE64_STANDARD);
        zstd::stream::copy_encode(data.as_slice(), &mut encoder, 0)
            .context("compressing tensor")
            .unwrap();

        let encoded = encoder.finish().context("converting to base64").unwrap();
        let result: Result<()> = async {
            let response = retry_async_operation(
                || {
                    reqwest_inject_trace_headers(reqwest::Client::new().put(put_url.as_str()))
                        .json(&json!({
                                "tensor": str::from_utf8(&encoded).expect("base64, so valid UTF8")
                        }))
                        .send()
                },
                || format!("storing {run_id}/{key}"),
            )
            .await
            .with_context(|| format!("calling store for {run_id}/{key}"))?;

            if let Err(err) = response.error_for_status_ref() {
                let status = response.status();
                let body = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read body".to_string());
                error!("storing {run_id}/{key}: HTTP {status} - {body}");
                return Err(err)
                    .with_context(|| format!("storing {run_id}/{key}: HTTP {status} - {body}"));
            }

            response
                .error_for_status()
                .with_context(|| format!("storing {run_id}/{key}"))
                .map(|_| ())
        }
        .await;
        res_tx.send(result).expect("sending back put result");
    }
}

impl<K> Drop for RemoteClient<K> {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Kill);
        let _ = self
            .worker_handle
            .take()
            .expect("RemoteClient must have a worker")
            .join();
    }
}
