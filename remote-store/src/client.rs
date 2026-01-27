use crate::{
    common::{RESPONSE_GET_BYTES, RESPONSE_PUT_BYTES, Request, ResponseGet, ResponsePut},
    metrics,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
    io,
    net::{SocketAddr, ToSocketAddrs},
    sync::Arc,
    thread::{self, JoinHandle},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    select,
    sync::{Mutex, mpsc, oneshot},
    time::timeout,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

const SERVER_TASK_RETRIES: usize = 2;

#[derive(Debug)]
pub struct Client<K> {
    cmd_tx: mpsc::UnboundedSender<Cmd<K>>,
    worker_handle: Option<JoinHandle<()>>,
    /// Storage keys that are currently being sent to server
    sending: HashSet<K>,
}

pub trait LocalStore {
    type Error: Display;
    type Key;

    /// Returns true if the store contains the given key.
    fn contains(&mut self, storage_key: &Self::Key) -> bool;

    /// Fetch the data associated with the given key.
    fn fetch(&mut self, storage_key: Self::Key) -> Result<&Vec<u8>, Self::Error>;

    /// Store the data under the given key.
    fn store(&mut self, storage_key: Self::Key, data: Vec<u8>) -> Result<(), Self::Error>;

    fn clean_up(&mut self, run_id: Uuid) -> Result<(), Self::Error>;
}

#[derive(Debug)]
struct ClientWorker<L, K> {
    local_store: L,
    server_addr: SocketAddr,
    cmd_rx: mpsc::UnboundedReceiver<Cmd<K>>,
    /// Storage keys that are currently being prefetched. A call to get the same
    /// key will attach a subscriber to it.
    prefetching: HashMap<K, Option<GetSubTx>>,
    prefetch_sub_rx: mpsc::UnboundedReceiver<Prefetched<K>>,
    prefetch_sub_tx: mpsc::UnboundedSender<Prefetched<K>>,
    monitor: TaskMonitor,
}

type TaskMonitor = Arc<Mutex<metrics::TaskMonitor<anyhow::Result<()>>>>;

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

impl<K> Client<K>
where
    K: 'static + Clone + Hash + Eq + Display + Send + Sync,
{
    pub fn new<L>(server_addr: impl ToSocketAddrs, local_store: L) -> Result<Self>
    where
        L: 'static + LocalStore + Send,
        L: LocalStore<Key = K>,
    {
        let server_addr = server_addr
            .to_socket_addrs()
            .context("Parsing address of a server")?
            .next()
            .context("Parsing address of a server")?;
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (prefetch_sub_tx, prefetch_sub_rx) = mpsc::unbounded_channel();

        let mut worker = ClientWorker {
            local_store,
            server_addr,
            cmd_rx,
            prefetching: HashMap::new(),
            prefetch_sub_rx,
            prefetch_sub_tx,
            monitor: Arc::new(Mutex::new(metrics::TaskMonitor::new())),
        };
        let async_rt = tokio::runtime::Runtime::new().context("Start client async runtime")?;
        let worker_handle = thread::spawn(move || async_rt.block_on(worker.run()));

        Ok(Client {
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
                // sender (owned by `Client`) is also dropped.
                Some(prefetched) = self.prefetch_sub_rx.recv() => {
                    let Prefetched { storage_key, result } = prefetched;
                    // Send the result to a subscriber if any attached
                    if let Some(sub) = self.prefetching.remove(&storage_key).expect("Prefetching in-progress must have a map entry") {
                        // If the rx is closed there's nothing left to do
                        let _ = sub.send(result);
                    } else if let Ok(data) = result {
                        if let Err(err) = self.local_store.store(storage_key, data){
                            error!("Failed to store prefetched data locally: {err}")
                        }
                    }
                },
                // Handle cmds from the client
                cmd = self.cmd_rx.recv() => {
                    match cmd {
                        None | Some(Cmd::Kill) => {
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
                task = Self::join_next_task(self.monitor.clone()) => {
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
        monitor: TaskMonitor,
    ) -> Option<(
        Result<anyhow::Result<()>, tokio::task::JoinError>,
        Cow<'static, str>,
    )> {
        let mut monitor = monitor.lock().await;
        // Timeout to avoid dead-locks from tasks spawning more tasks
        if let Ok(result) = timeout(Duration::from_millis(50), monitor.join_next()).await {
            return result;
        }
        None
    }

    async fn prefetch(&mut self, run_id: Uuid, storage_key: K) {
        if self.local_store.contains(&storage_key) {
            return;
        }

        if !self.prefetching.contains_key(&storage_key) {
            self.prefetching.insert(storage_key.clone(), None);

            let server_addr = self.server_addr;
            let prefetch_sub_tx = self.prefetch_sub_tx.clone();
            self.monitor
                .lock()
                .await
                .spawn(format!("prefetch/{run_id}/{storage_key}").into(), async move {
                    let mut retries = SERVER_TASK_RETRIES;
                    let result = loop {
                        let result =
                            Self::get_from_server(server_addr, run_id, storage_key.clone()).await;

                        if let Err(err) = &result {
                            error!("Failed to prefetch storage_key {storage_key} with {err:?}");
                        } else {
                            debug!("Prefetched {storage_key} from server");
                            break result;
                        }
                        if retries == 0 {
                            break result;
                        }
                        info!(
                            "Retrying to prefetch storage_key {storage_key} {retries} more time(s)"
                        );
                        retries -= 1;
                    };

                    let prefetched = Prefetched {
                        storage_key,
                        result,
                    };
                    // If the rx is closed there's nothing left to do
                    let _ = prefetch_sub_tx.send(prefetched);
                    Ok(())
                })
                .expect("to spawn prefetch task");
        } else {
            warn!("Duplicate prefetch call for {storage_key}");
        }
    }

    async fn clean_up(&mut self, run_id: Uuid) {
        let _ = self.local_store.clean_up(run_id);

        let server_addr = self.server_addr;
        // A task to wait for transfer from server
        self.monitor
            .lock()
            .await
            .spawn(format!("clean-up/{run_id}").into(), async move {
                // Open connection to server
                let mut stream = TcpStream::connect(server_addr)
                    .await
                    .context("Connecting to store server")?;

                // Send the request to the server
                let req = Request::CleanUp { run_id };
                let req_bytes = serde_json::to_vec(&req).context("Encoding request")?;
                stream
                    .write_all(&req_bytes)
                    .await
                    .context("Sending request")?;
                stream.flush().await.context("Flushing response")?;
                Ok(())
            })
            .expect("to spawn clean-up task");
    }

    async fn get(
        &mut self,
        run_id: Uuid,
        storage_key: &K,
        res_tx: oneshot::Sender<Result<Vec<u8>>>,
    ) {
        // Try fetch from local, fallback to remote if not found
        if let Ok(data) = self.local_store.fetch(storage_key.clone()).cloned() {
            res_tx.send(Ok(data)).expect("Sending GET result to client");
            return;
        }

        // Check if the key is being prefetched
        if let Some(sub) = self.prefetching.get_mut(storage_key) {
            if sub.is_some() {
                res_tx
                    .send(Err(anyhow!(
                        "The storage_key {storage_key} is already being received from another call"
                    )))
                    .expect("Sending GET result to client");
                return;
            }

            // This storage is being prefetched. Subscribe to receive its result.
            let (sub_tx, sub_rx) = oneshot::channel();
            *sub = Some(sub_tx);

            // A task to wait for prefetch to finish
            self.monitor
                .lock()
                .await
                .spawn(
                    format!("get-wait-prefetch/{run_id}/{storage_key}").into(),
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

        let server_addr = self.server_addr;
        let storage_key = storage_key.clone();
        // A task to wait for transfer from server
        self.monitor
            .lock()
            .await
            .spawn(format!("get/{run_id}/{storage_key}").into(), async move {
                let mut retries = SERVER_TASK_RETRIES;
                let result = loop {
                    let result =
                        Self::get_from_server(server_addr, run_id, storage_key.clone()).await;

                    if let Err(err) = &result {
                        error!("Failed to get storage_key {storage_key} with {err:?}");
                    } else {
                        debug!("Finished receiving {storage_key} from server");
                        break result;
                    }
                    if retries == 0 {
                        break result;
                    }
                    info!("Retrying to get storage_key {storage_key} {retries} more time(s)");
                    retries -= 1;
                };

                // Send back the result to the client
                res_tx.send(result).expect("Sending GET result to client");
                Ok(())
            })
            .expect("to spawn get task");
    }

    async fn get_from_server(
        server_addr: SocketAddr,
        run_id: Uuid,
        storage_key: K,
    ) -> Result<Vec<u8>> {
        // Open connection to server
        let mut stream = TcpStream::connect(server_addr)
            .await
            .context("Connecting to store server")?;

        // Send the request to the server
        let req = Request::Get {
            run_id,
            storage_key: storage_key.to_string(),
        };
        let req_bytes = serde_json::to_vec(&req).context("Encoding request")?;
        stream
            .write_all(&req_bytes)
            .await
            .context("Sending request")?;
        stream.flush().await.context("Flushing response")?;

        // Wait for the response
        let res_bytes = loop {
            // Wait for the socket to be readable
            stream
                .readable()
                .await
                .context("Waiting to read response")?;
            let mut buf = [0; RESPONSE_GET_BYTES];

            // Try to read data, this may still fail with `WouldBlock`
            // if the readiness event is a false positive.
            match stream.try_read(&mut buf) {
                Ok(n) => {
                    ensure!(
                        n == RESPONSE_GET_BYTES,
                        "Expected {RESPONSE_GET_BYTES}, but got {n}"
                    );
                    break buf;
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    continue;
                }
                Err(e) => {
                    bail!("Failed to get response from server with: {e}");
                }
            }
        };
        let ResponseGet { data_len } = ResponseGet::from_bytes(res_bytes);

        // Start receiving from the server
        let mut buf = vec![0; data_len as usize];
        let read_bytes = stream
            .read_exact(&mut buf)
            .await
            .context("Receiving requested file from server")?;
        debug_assert_eq!(read_bytes, data_len as usize);

        Ok(buf)
    }

    async fn put(
        &mut self,
        run_id: Uuid,
        storage_key: &K,
        data: Vec<u8>,
        res_tx: oneshot::Sender<Result<()>>,
    ) {
        let server_addr = self.server_addr;
        let storage_key = storage_key.to_string();
        // A task to transfer to server
        let monitor_clone = self.monitor.clone();
        self.monitor
            .lock()
            .await
            .spawn(format!("put/{run_id}/{storage_key}").into(), async move {
                let mut retries = SERVER_TASK_RETRIES;
                let mut reusable_data = Some(data);
                let result = loop {
                    let result = Self::put_aux(
                        monitor_clone.clone(),
                        server_addr,
                        run_id,
                        storage_key.clone(),
                        reusable_data.take().unwrap(),
                    )
                    .await;

                    match result {
                        Err((err, data)) => {
                            error!("Failed to put storage_key {storage_key} with {err:?}");
                            reusable_data = data;
                            if reusable_data.is_none() || retries == 0 {
                                break Err(err);
                            }
                            info!(
                                "Retrying to put storage_key {storage_key} {retries} more time(s)"
                            );
                            retries -= 1;
                        }
                        Ok(result) => {
                            debug!("Finished putting {storage_key} to server");
                            break Ok(result);
                        }
                    }
                };

                // Send back the result to the client
                res_tx.send(result).expect("Sending PUT result to client");
                Ok(())
            })
            .expect("to spawn put task");
    }

    /// Send put request to the server. On an error that's retriable the data is returned back to the caller.
    async fn put_aux(
        monitor: TaskMonitor,
        server_addr: SocketAddr,
        run_id: Uuid,
        storage_key: String,
        data: Vec<u8>,
    ) -> std::result::Result<(), (anyhow::Error, Option<Vec<u8>>)> {
        // Try to unwrap the result. If the result is an error attach data to it
        macro_rules! try_or_attach_data {
            ($res:expr) => {
                match $res {
                    Ok(result) => result,
                    Err(err) => return Err((err, Some(data))),
                }
            };
        }

        // Open connection to server
        let mut stream = try_or_attach_data!(
            TcpStream::connect(server_addr)
                .await
                .context("Connecting to store server")
        );

        // Send the request to the server
        let data_len = data.len() as u64;
        let task_name = format!("put-transfer-{storage_key}").into();
        let req = Request::Put {
            run_id,
            storage_key,
            data_len,
        };
        let req_bytes = try_or_attach_data!(serde_json::to_vec(&req).context("Encoding request"));
        try_or_attach_data!(
            stream
                .write_all(&req_bytes)
                .await
                .context("Sending request")
        );
        try_or_attach_data!(stream.flush().await.context("Flushing response"));

        // Wait for the response
        let res_bytes = loop {
            // Wait for the socket to be readable
            try_or_attach_data!(stream.readable().await.context("Waiting to read response"));
            let mut buf = [0; RESPONSE_PUT_BYTES];

            // Try to read data, this may still fail with `WouldBlock`
            // if the readiness event is a false positive.
            match stream.try_read(&mut buf) {
                Ok(n) => {
                    if n != RESPONSE_PUT_BYTES {
                        return Err((
                            anyhow!("Expected {RESPONSE_PUT_BYTES}, but got {n}"),
                            Some(data),
                        ));
                    }
                    break buf;
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    continue;
                }
                Err(e) => {
                    return Err((
                        anyhow!("Failed to get response from server with: {e}"),
                        Some(data),
                    ));
                }
            }
        };
        let _: ResponsePut = try_or_attach_data!(
            ResponsePut::try_from_bytes(&res_bytes).context("Decoding response for put request")
        );

        // Spawn a task for blocking IO to not block async runtime.
        // Any error from the task is not retriable as the task takes ownership of the data
        match monitor
            .lock()
            .await
            .spawn_blocking(task_name, move || {
                let mut stream = stream.into_std().context("Converting TcpStream")?;
                stream
                    .set_nonblocking(false)
                    .context("Setting stream to be non-blocking")?;

                let sent_bytes = io_copy::copy(&mut data.as_slice(), &mut stream)
                    .context("PUTing data onto server")?;
                if sent_bytes != data_len {
                    bail!("Expected to send {data_len}, but only sent {sent_bytes}");
                }
                Ok(())
            })
            .await
            .context("Spawning task to PUT data onto server")
        {
            Ok(_handle) => Ok(()),
            Err(err) => Err((err, None)),
        }
    }
}

impl<K> Drop for Client<K> {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Kill);
        let _ = self
            .worker_handle
            .take()
            .expect("Client must have a worker")
            .join();
    }
}
