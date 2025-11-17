use anyhow::{Context, ensure};
use deep_prove::store::MemStore;
use futures::StreamExt;
use memmap2::Mmap;
use object_store::{ObjectStore, path::Path as ObjectPath};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{fs as tokio_fs, io::AsyncWriteExt};
use tracing::info;

use crate::{S3Args, S3Store, StoreKind, store};

pub mod http;

pub struct WorkerResources {
    pub store: StoreKind,
    pub model_fetcher: ModelFetcher,
}

pub fn instantiate_store(
    args: &S3Args,
    model_cache_dir: PathBuf,
) -> anyhow::Result<WorkerResources> {
    fs::create_dir_all(&model_cache_dir)
        .with_context(|| format!("creating model cache root {}", model_cache_dir.display()))?;

    let models_bucket = args.s3_models_bucket.trim().to_string();

    let s3_config = S3ClientConfig::try_from_args(args);
    let model_client = s3_config
        .for_bucket(&models_bucket)
        .context("creating S3 client for models bucket")?;
    let model_fetcher =
        ModelFetcher::new(model_cache_dir.clone(), models_bucket.clone(), model_client);

    let store = if let Some(bucket) = args.s3_params_bucket.as_deref() {
        let client = s3_config
            .for_bucket(bucket)
            .context("creating S3 store client")?;
        let s3 = if args.fs_cache {
            S3Store::from(client).with_fs_cache(args.fs_cache_dir.clone())
        } else {
            S3Store::from(client)
        };
        info!("using S3 store");
        StoreKind::S3(s3)
    } else {
        info!("using in-memory store");
        StoreKind::Mem(MemStore::default())
    };

    Ok(WorkerResources {
        store,
        model_fetcher,
    })
}

#[derive(Clone)]
struct S3ClientConfig {
    region: String,
    endpoint: String,
    timeout: Duration,
    access_key_id: String,
    secret_access_key: String,
}

impl S3ClientConfig {
    fn try_from_args(args: &S3Args) -> Self {
        Self {
            region: args.s3_region.clone(),
            endpoint: args.s3_endpoint.clone(),
            timeout: Duration::from_secs(args.s3_timeout_secs),
            access_key_id: args.s3_access_key_id.clone(),
            secret_access_key: args.s3_secret_access_key.clone(),
        }
    }

    fn for_bucket(&self, bucket: &str) -> anyhow::Result<store::AmazonS3> {
        store::AmazonS3Builder::new()
            .with_region(self.region.clone())
            .with_bucket_name(bucket.to_string())
            .with_access_key_id(self.access_key_id.clone())
            .with_secret_access_key(self.secret_access_key.clone())
            .with_endpoint(self.endpoint.clone())
            .with_client_options(
                store::ClientOptions::default()
                    .with_timeout(self.timeout)
                    .with_allow_http(true),
            )
            .build()
            .context("building AWS S3 client")
    }
}

#[derive(Clone)]
pub struct ModelFetcher {
    model_cache_dir: PathBuf,
    bucket: String,
    client: store::AmazonS3,
}

impl ModelFetcher {
    fn new(model_cache_dir: PathBuf, bucket: String, client: store::AmazonS3) -> Self {
        Self {
            model_cache_dir,
            bucket,
            client,
        }
    }

    pub async fn fetch(&self, model_path: &str) -> anyhow::Result<Mmap> {
        let key = model_path.trim();
        ensure!(!key.is_empty(), "model path is empty");

        let cache_path = self.model_cache_dir.join(&self.bucket).join(Path::new(key));

        if tokio_fs::try_exists(&cache_path).await.unwrap_or(false) {
            return Self::map_file(&cache_path).await;
        }

        self.download_to_cache(key, &cache_path).await?;
        Self::map_file(&cache_path).await
    }

    async fn download_to_cache(&self, key: &str, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            tokio_fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating model cache dir {}", parent.display()))?;
        }

        let mut file = tokio_fs::File::create(path)
            .await
            .with_context(|| format!("creating cache file {}", path.display()))?;
        let mut stream = self
            .client
            .get(&ObjectPath::from(key.to_string()))
            .await
            .context("fetching model from S3")?
            .into_stream();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("streaming model bytes from S3")?;
            file.write_all(&bytes)
                .await
                .context("writing cached model chunk")?;
        }

        file.flush().await.context("flushing cached model")?;
        Ok(())
    }

    async fn map_file(path: &Path) -> anyhow::Result<Mmap> {
        let file =
            File::open(path).with_context(|| format!("opening cached model {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file).context("mmap-ing cached model") }?;
        Ok(mmap)
    }
}
