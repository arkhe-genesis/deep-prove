use crate::{
    StorageKey, StoreError,
    genstore::{local, local::LocalStore},
};
use anyhow::Result;
use remote_store::client as remote;
use serde::{Deserialize, Serialize};
use std::{net::ToSocketAddrs, path::Path};
use uuid::Uuid;

#[derive(Debug)]
pub struct Client {
    /// Remote store client
    remote: remote::Client<local::InternalKey>,
}

impl Client {
    pub fn new<P>(root: P, max_cache_size: usize, server_addr: impl ToSocketAddrs) -> Result<Self>
    where
        P: 'static + AsRef<Path> + Send,
    {
        let local = LocalStore::new(root, max_cache_size)?;
        let remote = remote::Client::new(server_addr, local)?;
        Ok(Self { remote })
    }

    pub(crate) fn prefetch<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
    ) -> Result<(), StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        self.remote
            .prefetch(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, storage_key),
            )
            .map_err(StoreError::RemoteStoreError)
    }

    pub(crate) fn fetch<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
    ) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        let bytes = self
            .remote
            .get(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, storage_key),
            )
            .map_err(StoreError::RemoteStoreError)?;
        let data: T = rmp_serde::from_slice(&bytes).map_err(StoreError::DeserializationError)?;
        Ok(data)
    }

    pub(crate) fn store<T>(
        &mut self,
        run_id: Uuid,
        storage_key: &StorageKey<T>,
        data: &T,
    ) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        let serialized = rmp_serde::to_vec(&data).map_err(StoreError::from)?;
        self.remote
            .put(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, storage_key),
                serialized,
            )
            .map_err(StoreError::RemoteStoreError)
    }

    pub(crate) fn clean_up(&self, run_id: Uuid) -> Result<(), StoreError> {
        self.remote
            .clean_up(run_id)
            .map_err(StoreError::RemoteStoreError)
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs,
        net::{IpAddr, Ipv4Addr, SocketAddr},
        path::PathBuf,
        sync::Arc,
        time::{Duration, Instant},
    };

    use crate::genstore::local::InternalKey;

    use super::*;
    use remote_store::{metrics::TaskMonitor, server};
    use tempfile::{TempDir, tempdir};
    use test_log::test;
    use tokio::{net::TcpListener, runtime::Runtime, sync::Mutex, task::JoinHandle};

    struct Test {
        server_state: Arc<server::AppState>,
        server_dir: TempDir,
        client: Client,
        _rt: Runtime,
        _server: JoinHandle<()>,
    }

    const BIG_DATA_SIZE: usize = 10 * 1024 * 1024;

    #[test]
    fn fetch() {
        let mut test = setup();
        let run_id = Uuid::new_v4();

        let data_0_key = StorageKey::<Vec<u8>>::new("test");
        let mut data_0_val = [1_u8; BIG_DATA_SIZE].to_vec();
        data_0_val[0] = 2;
        data_0_val[BIG_DATA_SIZE - 1] = 3;
        // Write the data to server's expected file path
        let data_0_path = test.server_file_path(run_id, &data_0_key);
        fs::create_dir(data_0_path.parent().unwrap()).unwrap();
        fs::write(&data_0_path, rmp_serde::to_vec_named(&data_0_val).unwrap()).unwrap();

        let data_1_key = StorageKey::<u8>::new("123");
        let data_1_val = 42_u8;
        // Write the data to server's expected file path
        let data_1_path = test.server_file_path(run_id, &data_1_key);
        fs::write(&data_1_path, rmp_serde::to_vec_named(&data_1_val).unwrap()).unwrap();

        let got_0 = test.client.fetch(run_id, &data_0_key).unwrap();
        assert_eq!(got_0, data_0_val);
        let got_1 = test.client.fetch(run_id, &data_1_key).unwrap();
        assert_eq!(got_1, data_1_val);
    }

    #[test]
    fn store() {
        let mut test = setup();
        let run_id = Uuid::new_v4();

        let data_0_key = StorageKey::<Vec<u8>>::new("test");
        let mut data_0_val = [1_u8; BIG_DATA_SIZE].to_vec();
        data_0_val[0] = 2;
        data_0_val[BIG_DATA_SIZE - 1] = 3;

        let data_1_key = StorageKey::<u8>::new("123");
        let data_1_val = 42_u8;

        let data_0_path = test.server_file_path(run_id, &data_0_key);
        let data_1_path = test.server_file_path(run_id, &data_1_key);

        test.client.store(run_id, &data_0_key, &data_0_val).unwrap();
        test.client.store(run_id, &data_1_key, &data_1_val).unwrap();

        // Wait for server to write received data to file
        let mut before_timeout = Duration::from_millis(200);
        let mut now = Instant::now();
        loop {
            if let Ok(monitor) = test.server_state.monitor.try_lock() {
                if monitor.is_empty() {
                    break;
                }
            }
            let delta = Instant::now() - now;
            if before_timeout <= delta {
                panic!("timed out waiting for async tasks to finish");
            }
            before_timeout -= delta;
            now = Instant::now();
        }

        // Check that the server has written the files
        assert!(data_0_path.exists());
        assert_eq!(
            rmp_serde::from_slice::<Vec<u8>>(&fs::read(&data_0_path).unwrap()).unwrap(),
            data_0_val
        );
        assert!(data_1_path.exists());
        assert_eq!(
            rmp_serde::from_slice::<u8>(&fs::read(&data_1_path).unwrap()).unwrap(),
            data_1_val
        );
    }

    #[test]
    fn prefetch() {
        let mut test = setup();
        let run_id = Uuid::new_v4();

        let data_0_key = StorageKey::<Vec<u8>>::new("test");
        let mut data_0_val = [1_u8; BIG_DATA_SIZE].to_vec();
        data_0_val[0] = 2;
        data_0_val[BIG_DATA_SIZE - 1] = 3;
        // Write the data to server's expected file path
        let data_0_path = test.server_file_path(run_id, &data_0_key);
        fs::create_dir(data_0_path.parent().unwrap()).unwrap();
        fs::write(&data_0_path, rmp_serde::to_vec_named(&data_0_val).unwrap()).unwrap();

        let data_1_key = StorageKey::<u8>::new("123");
        let data_1_val = 42_u8;
        // Write the data to server's expected file path
        let data_1_path = test.server_file_path(run_id, &data_1_key);
        fs::write(&data_1_path, rmp_serde::to_vec_named(&data_1_val).unwrap()).unwrap();

        // Start pre-fetching
        test.client.prefetch(run_id, &data_0_key).unwrap();
        test.client.prefetch(run_id, &data_1_key).unwrap();

        // Now fetch them - it should either get it from local if already received or attach a subscriber that waits for the prefetch to finish
        let got_0 = test.client.fetch(run_id, &data_0_key).unwrap();
        assert_eq!(got_0, data_0_val);
        let got_1 = test.client.fetch(run_id, &data_1_key).unwrap();
        assert_eq!(got_1, data_1_val);
    }

    fn setup() -> Test {
        let client_dir = tempdir().unwrap();
        let server_dir = tempdir().unwrap();
        let socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0);
        let rt = Runtime::new().unwrap();

        let server_listener = rt.block_on(async move { TcpListener::bind(socket).await.unwrap() });
        let server_port = server_listener.local_addr().unwrap().port();
        let server_dir_path = server_dir.path().to_path_buf();
        let server_state = Arc::new(server::AppState {
            store_dir: server_dir_path,
            monitor: Mutex::new(TaskMonitor::new()),
        });
        let server_state_clone = server_state.clone();
        let _server = rt.spawn(async move {
            server::run_on(server_listener, server_state_clone)
                .await
                .unwrap()
        });

        let server_addr = format!("127.0.0.1:{server_port}");
        let client = Client::new(client_dir, 10 * 1024 * 1024, server_addr).unwrap();

        Test {
            server_state,
            server_dir,
            client,
            _rt: rt,
            _server,
        }
    }

    impl Test {
        fn server_file_path<T>(&self, run_id: Uuid, storage_key: &StorageKey<T>) -> PathBuf {
            server::file_path(
                self.server_dir.path(),
                run_id,
                &InternalKey::from_storage_key_with_run_id(run_id, storage_key).to_string(),
            )
        }
    }
}
