use anyhow::Context;
use memmap2::Mmap;
use object_store::path::Path as ObjectPath;
use std::{
    fs::{self, File},
    path::PathBuf,
    time::Duration,
};
use tracing::info;

use crate::{S3Args, store};

pub mod http;
mod proving;

pub struct WorkerResources {
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
    let model_fetcher = ModelFetcher::new(model_cache_dir, model_client);

    Ok(WorkerResources { model_fetcher })
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
    client: store::AmazonS3,
}

impl ModelFetcher {
    fn new(model_cache_dir: PathBuf, client: store::AmazonS3) -> Self {
        Self {
            model_cache_dir,
            client,
        }
    }

    /// Fetch a graph context from S3 with local disk caching and mmap.
    /// Graph contexts are stored at `_graph/context/{model_hash}/{max_context}` in the models bucket.
    /// Returns an Mmap for zero-copy access to the context bytes.
    pub async fn fetch_graph_context_mmap(&self, graph_ctx_key: &str) -> anyhow::Result<Mmap> {
        let cache_path = self.model_cache_dir.join(graph_ctx_key.replace('/', "-"));

        // Handles caching, checksum validation, retries and resume download upon failure.
        store::download_object(
            &self.client,
            &ObjectPath::from(graph_ctx_key.to_string()),
            &cache_path,
            graph_ctx_key,
        )
        .await?;

        let mmap = unsafe {
            Mmap::map(
                &File::open(&cache_path)
                    .with_context(|| format!("opening {}", cache_path.display()))?,
            )
        }
        .with_context(|| format!("mmap-ing graph context {}", cache_path.display()))?;

        info!(
            "mmap'd graph context {} ({} bytes)",
            graph_ctx_key,
            mmap.len()
        );

        Ok(mmap)
    }
}
