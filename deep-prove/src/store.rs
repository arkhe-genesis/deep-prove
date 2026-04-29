//! PPs and scaled models KV storage.
#![allow(clippy::manual_async_fn)]

use anyhow::{Context, bail};
use ark_bn254::Bn254;
use dp_crypto::arkyper::HyperKZG;
use exponential_backoff::Backoff;
use futures::StreamExt;
use memmap2::Mmap;
use object_store::{Attribute, GetOptions, GetRange, ObjectStore, PutPayload, path::Path};
#[doc(inline)]
pub use object_store::{
    ClientOptions,
    aws::{AmazonS3, AmazonS3Builder},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::HashMap,
    env,
    fmt::Debug,
    fs,
    fs::File,
    future::Future,
    os::unix::io::AsRawFd,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};
use tempfile::TempDir;
use tokio::{
    io::{AsyncWriteExt, BufWriter},
    time::sleep,
};
use tracing::{debug, error, info, warn};
use zkml::{
    Element, ProverContext,
    iop::context::VerifierContext,
    model::Model,
    quantization::{ModelMetadata, ScalingStrategyKind},
};

#[derive(Debug, Clone)]
pub struct ParamsKey {
    pub model_file_hash: String,
}

#[derive(Debug, Clone)]
pub struct ModelKey {
    pub model_file_hash: String,
    pub scaling_strategy: ScalingStrategyKind,
    pub scaling_input_hash: Option<String>,
}

type F = ark_bn254::Fr;
type Pcs = HyperKZG<Bn254>;

#[derive(Serialize, Deserialize)]
pub struct Params {
    pub prover: ProverContext<'static, F, Pcs>,
    pub verifier: VerifierContext<F, Pcs>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ScaledModel {
    pub model: Model<Element>,
    pub model_metadata: ModelMetadata,
}

pub trait Store: Clone {
    /// Try to get the params from store.
    fn get_params(
        &mut self,
        key: &ParamsKey,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send;

    /// Store the params.
    fn insert_params(
        &mut self,
        key: &ParamsKey,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Try to get the model from store. If not present, initialize the value with the given function, store it and return.
    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: &ModelKey,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send;
}

/// AWS S3 store for prod.
#[derive(Clone, derive_more::From)]
pub struct S3Store {
    store: AmazonS3,
    fs_cache: Option<Arc<TempDir>>,
}

#[derive(Debug, Clone)]
struct RemoteObjectMetadata {
    size: u64,
    etag: Option<String>,
    sha256: Option<String>,
}

impl From<AmazonS3> for S3Store {
    fn from(store: AmazonS3) -> Self {
        S3Store {
            store,
            fs_cache: None,
        }
    }
}

impl S3Store {
    pub fn with_fs_cache(mut self, fs_cache_dir: PathBuf) -> Self {
        self.fs_cache = Some(Arc::new(
            TempDir::new_in(fs_cache_dir).expect("able to setup an S3 store cache in a temp dir"),
        ));
        self
    }
}

impl Store for S3Store {
    fn get_params(
        &mut self,
        key: &ParamsKey,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send {
        async move {
            let key = params_key(key);
            let S3Store { store, fs_cache } = self;

            // Try read from FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));
            if let Some(path) = &cache_path
                && tokio::fs::try_exists(&path)
                    .await
                    .context("access FS cache")?
            {
                let bytes = tokio::fs::read(path).await?;
                let value = serde_json::from_slice::<Params>(&bytes)
                    .context("decoding params value from FS cache")?;
                return Ok(Some(value));
            }
            match store.get(&key).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<Params>(&bytes)
                        .context("decoding params value from S3")?;

                    // Cache to FS
                    if let Some(path) = cache_path {
                        tokio::fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        tokio::fs::write(&path, &bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    Ok(Some(value))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(e) => {
                    bail!(e);
                }
            }
        }
    }

    fn insert_params(
        &mut self,
        key: &ParamsKey,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            let value_bytes: Vec<u8> =
                serde_json::to_vec(&params).context("serializing params to store")?;
            let key = params_key(key);
            let S3Store { store, fs_cache } = self;

            // Write to FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));

            if let Some(path) = cache_path
                && !tokio::fs::try_exists(&path)
                    .await
                    .context("access FS cache")?
            {
                tokio::fs::create_dir_all(&path)
                    .await
                    .context("create FS cache dirs")?;
                tokio::fs::write(&path, &value_bytes)
                    .await
                    .context("write params to FS cache")?;
            }

            if store
                .get_opts(
                    &key,
                    GetOptions {
                        head: true,
                        ..Default::default()
                    },
                )
                .await
                .is_ok()
            {
                bail!("trying to insert params with {key} that is already present")
            }
            store
                .put(&key, PutPayload::from(value_bytes))
                .await
                .context("storing generated params in S3 store")?;
            Ok(())
        }
    }

    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: &ModelKey,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send,
    {
        async move {
            let key = model_key(key);
            let S3Store { store, fs_cache } = self;

            // Try read from FS cache first
            let cache_path = fs_cache
                .as_ref()
                .map(|cache| cache.path().join(key.to_string()));
            if let Some(path) = &cache_path
                && tokio::fs::try_exists(&path)
                    .await
                    .context("access FS cache")?
            {
                let bytes = tokio::fs::read(path).await?;
                let value = serde_json::from_slice::<ScaledModel>(&bytes)
                    .context("decoding scaled model value from FS cache")?;
                return Ok(value);
            }

            match store.get(&key).await {
                Ok(result) => {
                    let bytes = result.bytes().await?;
                    let value = serde_json::from_slice::<ScaledModel>(&bytes)
                        .context("decoding scaled model value from S3")?;

                    // Cache to FS
                    if let Some(path) = cache_path {
                        tokio::fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        tokio::fs::write(&path, &bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    Ok(value)
                }
                Err(object_store::Error::NotFound { .. }) => {
                    let value = init().await?;
                    let value_bytes: Vec<u8> =
                        serde_json::to_vec(&value).context("serializing scaled model to store")?;

                    // Write to FS cache first
                    if let Some(path) = cache_path {
                        tokio::fs::create_dir_all(&path)
                            .await
                            .context("create FS cache dirs")?;
                        tokio::fs::write(&path, &value_bytes)
                            .await
                            .context("write params to FS cache")?;
                    }

                    store
                        .put(&key, PutPayload::from(value_bytes))
                        .await
                        .context("storing generated params in S3 store")?;
                    Ok(value)
                }
                Err(e) => {
                    bail!(e);
                }
            }
        }
    }
}

/// The number of attempts after which S3 download is considered failed.
const ATTEMPTS: u32 = 10;
/// The minimum waiting time before retrying an S3 download.
const MIN_WAIT_MS: u64 = 500;
/// The maximum waiting time before considering an S3 download as failed.
const MAX_WAIT_MS: u64 = 30_000;
/// Attempts to wait on per-file lock before giving up.
const DOWNLOAD_LOCK_ATTEMPTS: u32 = 10;
/// Wait between lock acquisition attempts.
const DOWNLOAD_LOCK_WAIT_SECS: u64 = 2;

/// Retry the given asynchronous operation with exponential backoff.
async fn retry_async_operation<
    const RETRY_ATTEMPTS: u32,
    const RETRY_MIN_WAIT_MS: u64,
    const RETRY_MAX_WAIT_MS: u64,
    F,
    Fut,
    T,
    E: Debug,
>(
    func: F,
    log: impl Fn() -> String,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    for (attempt_idx, duration) in Backoff::new(
        RETRY_ATTEMPTS,
        Duration::from_millis(RETRY_MIN_WAIT_MS),
        Duration::from_millis(RETRY_MAX_WAIT_MS),
    )
    .into_iter()
    .enumerate()
    {
        let result = func().await;
        match &result {
            Ok(_) => {
                if attempt_idx > 0 {
                    info!(
                        "operation succeeded after retries. operation: {} attempts: {}",
                        log(),
                        attempt_idx + 1
                    );
                }
                return result;
            }
            Err(err) => match duration {
                Some(duration) => {
                    warn!(
                        "failed to execute operation. operation: {} attempt: {}/{} retry_secs: {} err: {:?}",
                        log(),
                        attempt_idx + 1,
                        RETRY_ATTEMPTS,
                        duration.as_secs(),
                        err
                    );
                    sleep(duration).await;
                }
                None => {
                    error!(
                        "eventually failed to execute operation {} after {} attempts: {:?}",
                        log(),
                        RETRY_ATTEMPTS,
                        err
                    );
                    return result;
                }
            },
        }
    }

    unreachable!()
}

pub async fn download_object(
    client: &AmazonS3,
    object_path: &Path,
    cache_path: &FsPath,
    object_key: &str,
) -> anyhow::Result<()> {
    let remote = retry_async_operation::<ATTEMPTS, MIN_WAIT_MS, MAX_WAIT_MS, _, _, _, _>(
        || async {
            let head = client
                .get_opts(
                    object_path,
                    GetOptions {
                        head: true,
                        ..Default::default()
                    },
                )
                .await
                .with_context(|| format!("fetching metadata for {}", object_key))?;

            Ok::<_, anyhow::Error>(RemoteObjectMetadata {
                size: head.meta.size,
                etag: head.meta.e_tag,
                sha256: head
                    .attributes
                    .get(&Attribute::Metadata("sha256".into()))
                    .map(|value| value.as_ref().to_string()),
            })
        },
        || format!("s3_op: head: key {}", object_key),
    )
    .await?;
    if remote.etag.is_none() && remote.sha256.is_none() {
        bail!(
            "remote object {} has neither etag nor sha256 metadata for integrity validation",
            object_key
        );
    }
    debug!(
        "resolved remote object metadata for {} (size={} has_sha256={} has_etag={})",
        object_key,
        remote.size,
        remote.sha256.is_some(),
        remote.etag.is_some()
    );

    if cache_valid(cache_path, object_key, &remote)? {
        debug!("cache hit for {}, skipping download", object_key);
        return Ok(());
    }

    debug!(
        "cache miss or stale cache for {}, preparing resumable download",
        object_key
    );
    remove_stale_cache(cache_path)?;

    let lock_path = cache_path.with_extension("lock");
    let mut lock = None;
    for attempt in 0..DOWNLOAD_LOCK_ATTEMPTS {
        if cache_valid(cache_path, object_key, &remote)? {
            debug!(
                "cache for {} became valid while waiting for lock, skipping download",
                object_key
            );
            return Ok(());
        }

        // Acquire a lock using flock which the kernel releases when the process exits on
        // SIGKILL, OOM kill, etc. so stale lock files are not left behind
        match DownloadLock::try_acquire(&lock_path) {
            Ok(download_lock) => {
                debug!(
                    "acquired download lock {} for {}",
                    lock_path.display(),
                    object_key
                );
                lock = Some(download_lock);
                break;
            }
            Err(err) => {
                if attempt == 0 {
                    warn!("waiting for download lock {}: {}", lock_path.display(), err);
                }
                sleep(Duration::from_secs(DOWNLOAD_LOCK_WAIT_SECS)).await;
            }
        }
    }

    let _lock = lock.ok_or_else(|| {
        anyhow::anyhow!(
            "timed out waiting for download lock {}",
            lock_path.display()
        )
    })?;

    if cache_valid(cache_path, object_key, &remote)? {
        debug!(
            "cache for {} became valid after lock acquisition, skipping download",
            object_key
        );
        return Ok(());
    }

    let partial_path = cache_path.with_extension("partial");
    retry_async_operation::<ATTEMPTS, MIN_WAIT_MS, MAX_WAIT_MS, _, _, _, _>(
        || {
            download_object_once(
                client,
                object_path,
                object_key,
                cache_path,
                &partial_path,
                &remote,
            )
        },
        || format!("s3_op: download: key {}", object_key),
    )
    .await
}

fn cache_sidecar_paths(cache_path: &FsPath) -> (PathBuf, PathBuf) {
    (
        cache_path.with_extension("etag"),
        cache_path.with_extension("sha256"),
    )
}

fn read_sidecar(path: &FsPath) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn write_sidecar(path: &FsPath, value: &str) -> anyhow::Result<()> {
    fs::write(path, value).with_context(|| format!("writing {}", path.display()))
}

fn cache_valid(
    cache_path: &FsPath,
    object_key: &str,
    remote: &RemoteObjectMetadata,
) -> anyhow::Result<bool> {
    match fs::metadata(cache_path) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!("stat {}", cache_path.display())));
        }
    }

    let (etag_path, sha_path) = cache_sidecar_paths(cache_path);

    if let Some(expected_sha) = remote.sha256.as_ref() {
        if let Some(stored) = read_sidecar(&sha_path)
            && stored == *expected_sha
        {
            return Ok(true);
        }

        // Recompute digest when sidecar checksum is missing or stale.
        let file =
            File::open(cache_path).with_context(|| format!("opening {}", cache_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap-ing {}", cache_path.display()))?;
        let computed = format!("{:x}", Sha256::digest(&mmap));
        if computed == *expected_sha {
            write_sidecar(&sha_path, &computed)?;
            return Ok(true);
        }
        warn!("cache checksum mismatch for {}, redownloading", object_key);
        return Ok(false);
    }

    if let Some(expected_etag) = remote.etag.as_ref() {
        if let Some(stored) = read_sidecar(&etag_path) {
            return Ok(stored == *expected_etag);
        }
        write_sidecar(&etag_path, expected_etag)?;
        return Ok(true);
    }
    unreachable!("remote object must have etag or sha256 metadata")
}

fn remove_stale_cache(cache_path: &FsPath) -> anyhow::Result<()> {
    if cache_path.exists() {
        warn!("removing stale cache file {}", cache_path.display());
        fs::remove_file(cache_path)
            .with_context(|| format!("removing stale {}", cache_path.display()))?;
    }
    let (etag_path, sha_path) = cache_sidecar_paths(cache_path);
    let _ = fs::remove_file(&etag_path);
    let _ = fs::remove_file(&sha_path);
    Ok(())
}

/// Flock based file locking
///
/// The kernel releases the lock when the file descriptor is closed in both
/// `Drop` path and when the process is killed SIGKILL, OOM, etc. which eliminates stale lock files.
struct DownloadLock {
    /// Keeps the file descriptor open so the kernel holds the flock.
    _file: File,
    /// Path to the lock file.
    path: PathBuf,
}

impl DownloadLock {
    fn try_acquire(path: &FsPath) -> anyhow::Result<Self> {
        let file =
            File::create(path).with_context(|| format!("creating lock file {}", path.display()))?;

        let flock_result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if flock_result != 0 {
            let err = std::io::Error::last_os_error();
            anyhow::bail!("lock held by another process ({}): {}", path.display(), err);
        }

        Ok(Self {
            _file: file,
            path: path.to_path_buf(),
        })
    }
}

impl Drop for DownloadLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

async fn download_object_once(
    client: &AmazonS3,
    object_path: &Path,
    object_key: &str,
    cache_path: &FsPath,
    partial_path: &FsPath,
    remote: &RemoteObjectMetadata,
) -> anyhow::Result<()> {
    let mut hasher = Sha256::new();
    let mut start = 0u64;

    if partial_path.exists() {
        let partial_size = fs::metadata(partial_path)
            .with_context(|| format!("stat {}", partial_path.display()))?
            .len();
        if partial_size > remote.size {
            warn!(
                "partial file larger than remote object for {}, removing {}",
                object_key,
                partial_path.display()
            );
            fs::remove_file(partial_path)
                .with_context(|| format!("removing oversized {}", partial_path.display()))?;
        } else if partial_size > 0 {
            debug!(
                "resuming download for {} from offset {} of {} bytes",
                object_key, partial_size, remote.size
            );
            let partial = File::open(partial_path)
                .with_context(|| format!("opening {}", partial_path.display()))?;
            let mmap = unsafe { Mmap::map(&partial) }
                .with_context(|| format!("mmap-ing {}", partial_path.display()))?;
            hasher.update(&mmap);
            start = partial_size;
        }
    }

    // Skip download when the partial file already contains all expected bytes
    // this can happen if a previous attempt downloaded the object fully to .partial but failed before renaming it.
    if start < remote.size {
        let mut opts = GetOptions::default();
        if start > 0 {
            opts.range = Some(GetRange::from(start..remote.size));
        }
        // Use If-Match so S3 returns 412 Precondition Failed if the object was
        // replaced between the HEAD and this GET. This prevents corrupt resumptions
        // as without it, the new bytes would be appended to an old partial file.
        if let Some(etag) = &remote.etag {
            opts.if_match = Some(etag.clone());
        }

        let result = match client.get_opts(object_path, opts).await {
            Ok(res) => res,
            Err(object_store::Error::NotSupported { .. }) if start > 0 => {
                warn!(
                    "range download not supported for {}, restarting from byte 0",
                    object_key
                );
                fs::remove_file(partial_path).ok();
                start = 0;
                hasher = Sha256::new();
                let mut fallback_opts = GetOptions::default();
                if let Some(etag) = &remote.etag {
                    fallback_opts.if_match = Some(etag.clone());
                }
                client.get_opts(object_path, fallback_opts).await?
            }
            Err(err) => return Err(err.into()),
        };

        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(start > 0)
            .truncate(start == 0)
            .open(partial_path)
            .await
            .with_context(|| format!("opening {}", partial_path.display()))?;
        let mut file = BufWriter::new(file);

        let mut stream = result.into_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("receiving from S3")?;
            file.write_all(&chunk)
                .await
                .context("appending to file from S3")?;
            hasher.update(&chunk);
        }
        file.flush().await?;
    }

    // Verify the on disk size of in memory hash independently as the hasher is fed from memory buffers.
    let digest = format!("{:x}", hasher.finalize());
    if let Some(expected) = remote.sha256.as_ref()
        && &digest != expected
    {
        fs::remove_file(partial_path).ok();
        return Err(anyhow::anyhow!(
            "checksum mismatch for {} (expected {}, got {})",
            object_key,
            expected,
            digest
        ));
    }

    let cache_parent = cache_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("cache path has no parent"))?;
    fs::create_dir_all(cache_parent)
        .with_context(|| format!("creating model cache dir {}", cache_parent.display()))?;

    if cache_path.exists() {
        debug!(
            "cache file already exists for {}, dropping partial file {}",
            object_key,
            partial_path.display()
        );
        let _ = fs::remove_file(partial_path);
    } else {
        fs::rename(partial_path, cache_path)
            .with_context(|| format!("atomically moving object into {}", cache_path.display()))?;
    }

    let (etag_path, sha_path) = cache_sidecar_paths(cache_path);
    write_sidecar(&sha_path, &digest)?;
    if let Some(etag) = remote.etag.as_ref() {
        write_sidecar(&etag_path, etag)?;
    }

    info!("downloaded {} ({} bytes)", object_key, remote.size);
    Ok(())
}

/// In-memory store for testing.
#[derive(Clone, Default)]
pub struct MemStore {
    /// The [`Params`] are stored serialised.
    pps: Arc<Mutex<HashMap<Key, Vec<u8>>>>,
    models: Arc<Mutex<HashMap<Key, ScaledModel>>>,
}

#[derive(Clone, Default)]
pub struct MemStoreInner {}

impl Store for MemStore {
    fn get_params(
        &mut self,
        key: &ParamsKey,
    ) -> impl Future<Output = anyhow::Result<Option<Params>>> + Send {
        async move {
            let key = params_key(key);
            let guard = self.pps.lock().unwrap();
            guard
                .get(&key)
                .map_or(Ok(None), |bytes| {
                    serde_json::from_slice::<Params>(bytes).map(Some)
                })
                .map_err(anyhow::Error::from)
        }
    }

    fn insert_params(
        &mut self,
        key: &ParamsKey,
        params: Params,
    ) -> impl Future<Output = anyhow::Result<()>> + Send {
        async move {
            let key = params_key(key);
            let mut guard = self.pps.lock().unwrap();
            let bytes = serde_json::to_vec(&params).context("serializing params to store")?;
            guard.insert(key, bytes);
            Ok(())
        }
    }

    fn get_or_init_model_with<F, FR>(
        &mut self,
        key: &ModelKey,
        init: F,
    ) -> impl Future<Output = anyhow::Result<ScaledModel>> + Send
    where
        F: FnOnce() -> FR + Send,
        FR: Future<Output = anyhow::Result<ScaledModel>> + Send,
    {
        async move {
            let key = model_key(key);
            let get_result = {
                let guard = self.models.lock().unwrap();
                guard.get(&key).cloned()
            };
            let value = match get_result {
                Some(value) => value,
                None => {
                    let value = init().await?;
                    let mut guard = self.models.lock().unwrap();
                    guard.insert(key, value.clone());
                    value
                }
            };
            Ok(value)
        }
    }
}

type Key = Path;

#[derive(derive_more::Display)]
enum KeyKind {
    /// Proving parameters
    Params,
    /// Scaled model
    Model,
}

/// A store key for parameters
fn params_key(ParamsKey { model_file_hash }: &ParamsKey) -> Key {
    let prefix = KeyKind::Params.to_string();
    let prefix = prefix.as_str();
    let pkg_major_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .map(|version| Cow::from(version.major.to_string()))
        .unwrap_or_else(|_| Cow::from("version-unknown"));
    Path::from_iter([prefix, &pkg_major_version, model_file_hash])
}

/// A store key for a scaled model
fn model_key(
    ModelKey {
        model_file_hash,
        scaling_strategy,
        scaling_input_hash,
    }: &ModelKey,
) -> Key {
    let prefix = KeyKind::Model.to_string();
    let prefix = prefix.as_str();
    let scaling_strategy = scaling_strategy.to_string();
    let scaling_strategy = scaling_strategy.as_str();
    match scaling_input_hash {
        Some(scaling_input_hash) => Path::from_iter([
            prefix,
            model_file_hash,
            scaling_strategy,
            scaling_input_hash,
        ]),
        None => Path::from_iter([prefix, model_file_hash, scaling_strategy]),
    }
}
