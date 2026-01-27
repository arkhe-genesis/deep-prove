use std::{
    borrow::Cow,
    fs::File,
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{
    common::{Request, ResponseGet, ResponsePut},
    metrics::TaskMonitor,
};

use anyhow::{Context, anyhow, bail};
use tokio::{fs, io::AsyncWriteExt, net::TcpListener, select, sync::Mutex};
use tracing::{error, info};
use uuid::Uuid;

#[derive(Debug)]
pub struct AppState {
    pub store_dir: PathBuf,
    pub monitor: Mutex<TaskMonitor<anyhow::Result<()>>>,
}

impl AppState {
    /// Format FS path from the `run_id`
    fn run_dir(&self, run_id: Uuid) -> PathBuf {
        run_dir(&self.store_dir, run_id)
    }

    /// Format FS path from the `storage_key`
    fn file_path(&self, run_id: Uuid, storage_key: &str) -> PathBuf {
        file_path(&self.store_dir, run_id, storage_key)
    }

    /// Join next async task. Returns `None` if there are no tasks
    async fn join_next_task(
        &self,
    ) -> Option<(
        Result<anyhow::Result<()>, tokio::task::JoinError>,
        Cow<'static, str>,
    )> {
        let mut monitor = self.monitor.lock().await;
        monitor.join_next().await
    }
}

pub fn file_path(store_dir: &Path, run_id: Uuid, storage_key: &str) -> PathBuf {
    run_dir(store_dir, run_id).join(format!("{storage_key}.bin"))
}

fn run_dir(store_dir: &Path, run_id: Uuid) -> PathBuf {
    store_dir.join(run_id.to_string())
}

pub async fn run(store_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
    let listener = TcpListener::bind(socket)
        .await
        .context("Opening server TCP socket")?;

    let monitor = Mutex::new(TaskMonitor::new());
    let state = Arc::new(AppState { store_dir, monitor });

    run_on(listener, state).await
}

pub async fn run_on(listener: TcpListener, state: Arc<AppState>) -> anyhow::Result<()> {
    loop {
        select! {
            conn = listener.accept() => {
                match conn {
                    Ok((io, remote_addr)) => {
                        let state_clone = state.clone();
                        tokio::spawn(
                            async move {
                                let result = handle_connection(state_clone, io).await;

                                if let Err(err) = &result {
                                    error!("Request from {remote_addr} failed with: {err}");
                                }
                            },
                        );
                    }
                    Err(err) => error!("Failed to accept a connection with: {err}"),
                }
            },
            task = state.join_next_task() => {
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

async fn handle_connection(
    state: Arc<AppState>,
    stream: tokio::net::TcpStream,
) -> anyhow::Result<()> {
    let buf = loop {
        // Wait for the socket to be readable
        stream.readable().await.context("Waiting to read request")?;
        // Should fit a dynamically sized storage key within a request
        let mut buf = [0; 1024];

        // Try to read data, this may still fail with `WouldBlock`
        // if the readiness event is a false positive.
        match stream.try_read(&mut buf) {
            Ok(n) => {
                break buf.into_iter().take(n).collect::<Vec<_>>();
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                continue;
            }
            Err(e) => {
                bail!("Failed to get request from client with: {e}");
            }
        }
    };
    let req: Request = serde_json::from_slice(&buf).context("Decoding request")?;
    match req {
        Request::Get {
            run_id,
            storage_key,
        } => handle_get(state, stream, run_id, storage_key)
            .await
            .context("Get request"),
        Request::Put {
            run_id,
            storage_key,
            data_len,
        } => handle_put(state, stream, run_id, storage_key, data_len)
            .await
            .context("Put request"),
        Request::CleanUp { run_id } => handle_clean_up(state, run_id)
            .await
            .context("CleanUp request"),
    }
}

async fn handle_get(
    state: Arc<AppState>,
    mut stream: tokio::net::TcpStream,
    run_id: Uuid,
    storage_key: String,
) -> anyhow::Result<()> {
    info!("Get request for key {storage_key}");

    let path = state.file_path(run_id, &storage_key);
    let metadata = path
        .metadata()
        .context("Requested key {storage_key} is not present")?;
    let data_len = metadata.len();
    let mut file_desc =
        File::open(&path).with_context(|| format!("opening backing file for `{storage_key}`"))?;

    // Send the response with the data length
    let res = ResponseGet { data_len };
    let res_bytes = res.to_bytes();
    stream
        .write_all(&res_bytes)
        .await
        .context("Writing response")?;
    stream.flush().await.context("Flushing response")?;

    // Spawn a task with data transfer to not block async runtime with blocking IO
    let _handle = state
        .monitor
        .lock()
        .await
        .spawn_blocking(format!("get-transfer-{storage_key}").into(), move || {
            let mut stream = stream.into_std().context("Converting TcpStream")?;
            stream
                .set_nonblocking(false)
                .context("Setting stream to be non-blocking")?;
            let sent_bytes =
                io_copy::copy(&mut file_desc, &mut stream).context("Sending data to client")?;

            if sent_bytes != data_len {
                Err(anyhow!(
                    "Expected to send {data_len}, but only sent {sent_bytes}"
                ))?;
            }
            anyhow::Ok(())
        })
        .await
        .context("Spawning task to send data to client")?;

    Ok(())
}

async fn handle_put(
    state: Arc<AppState>,
    mut stream: tokio::net::TcpStream,
    run_id: Uuid,
    storage_key: String,
    data_len: u64,
) -> anyhow::Result<()> {
    info!("Put request for key {storage_key}");

    let path = state.file_path(run_id, &storage_key);
    let _ = fs::create_dir(&path.parent().context("creating parent path")?).await;
    let mut file_desc =
        File::create_new(&path).context("Creating file to store put request data")?;

    // Send the response that we're ready to receive
    let res = ResponsePut;
    let res_bytes = res.to_bytes();
    stream
        .write_all(&res_bytes)
        .await
        .context("Sending response")?;
    stream.flush().await.context("Flushing response")?;

    // Spawn a task for blocking IO to not block async runtime
    let _handle = state
        .monitor
        .lock()
        .await
        .spawn_blocking(format!("put-transfer-{storage_key}").into(), move || {
            let mut stream = stream.into_std().context("Converting TcpStream")?;
            stream
                .set_nonblocking(false)
                .context("Setting stream to be non-blocking")?;
            let recv_bytes =
                io_copy::copy(&mut stream, &mut file_desc).context("Sending data to client")?;

            if recv_bytes != data_len {
                Err(anyhow!(
                    "Expected to receive {data_len}, but only got {recv_bytes}"
                ))?;
            }

            anyhow::Ok(())
        })
        .await;

    Ok(())
}

async fn handle_clean_up(state: Arc<AppState>, run_id: Uuid) -> anyhow::Result<()> {
    let dir = state.run_dir(run_id);
    fs::remove_dir_all(dir)
        .await
        .with_context(|| format!("Removing stored run directory for run {run_id}"))
}
