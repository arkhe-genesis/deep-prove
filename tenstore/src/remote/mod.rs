use crate::{
    StorageKey, StoreError,
    local::{self, DiskStore},
    remote::client::RemoteClient,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use url::Url;
use uuid::Uuid;

mod client;
mod metrics;

#[derive(Debug)]
pub struct RemoteStore {
    /// Remote store client
    remote: RemoteClient<local::InternalKey>,
}

impl RemoteStore {
    pub fn new<P>(root: P, max_cache_size: usize, server_addr: Url) -> Result<Self>
    where
        P: 'static + AsRef<Path> + Send,
    {
        let local = DiskStore::new(root, max_cache_size)?;
        let remote = client::RemoteClient::new(server_addr, local)?;
        Ok(Self { remote })
    }

    pub(crate) fn prefetch<T>(
        &mut self,
        run_id: Uuid,
        key: &StorageKey<T>,
    ) -> Result<(), StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        self.remote
            .prefetch(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, key),
            )
            .map_err(StoreError::RemoteStoreError)
    }

    pub(crate) fn fetch<T>(&mut self, run_id: Uuid, key: &StorageKey<T>) -> Result<T, StoreError>
    where
        T: for<'a> Deserialize<'a>,
    {
        let bytes = self
            .remote
            .get(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, key),
            )
            .map_err(StoreError::RemoteStoreError)?;
        let data: T = rmp_serde::from_slice(&bytes).map_err(StoreError::DeserializationError)?;
        Ok(data)
    }

    pub(crate) fn store<T>(
        &mut self,
        run_id: Uuid,
        key: &StorageKey<T>,
        data: &T,
    ) -> Result<(), StoreError>
    where
        T: Serialize,
    {
        let serialized = rmp_serde::to_vec(&data).map_err(StoreError::from)?;
        self.remote
            .put(
                run_id,
                local::InternalKey::from_storage_key_with_run_id(run_id, key),
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
    use crate::{StorageKey, remote::RemoteStore};
    use test_log::test;
    use url::Url;
    use uuid::Uuid;

    struct Test {
        client: RemoteStore,
    }

    const BIG_DATA_SIZE: usize = 10 * 1024 * 1024;

    #[test]
    #[ignore = "Requires a running GW"]
    fn fetch() {
        let mut test = setup();
        let run_id = Uuid::new_v4();

        let data_0_key = StorageKey::<Vec<u8>>::new("test");
        let mut data_0_val = [1_u8; BIG_DATA_SIZE].to_vec();
        data_0_val[0] = 2;
        data_0_val[BIG_DATA_SIZE - 1] = 3;
        test.client.store(run_id, &data_0_key, &data_0_val).unwrap();

        let data_1_key = StorageKey::<u8>::new("123");
        let data_1_val = 42_u8;
        test.client.store(run_id, &data_1_key, &data_1_val).unwrap();

        let got_0 = test.client.fetch(run_id, &data_0_key).unwrap();
        assert_eq!(got_0, data_0_val);
        let got_1 = test.client.fetch(run_id, &data_1_key).unwrap();
        assert_eq!(got_1, data_1_val);

        test.client.clean_up(run_id).unwrap();
    }

    #[test]
    #[ignore = "Requires a running GW"]
    fn prefetch() {
        let mut test = setup();
        let run_id = Uuid::new_v4();

        let data_0_key = StorageKey::<Vec<u8>>::new("test");
        let mut data_0_val = [1_u8; BIG_DATA_SIZE].to_vec();
        data_0_val[0] = 2;
        data_0_val[BIG_DATA_SIZE - 1] = 3;
        test.client.store(run_id, &data_0_key, &data_0_val).unwrap();

        let data_1_key = StorageKey::<u8>::new("123");
        let data_1_val = 42_u8;
        test.client.store(run_id, &data_1_key, &data_1_val).unwrap();

        // Start pre-fetching
        test.client.prefetch(run_id, &data_0_key).unwrap();
        test.client.prefetch(run_id, &data_1_key).unwrap();

        // Now fetch them - it should either get it from local if already
        // received or attach a subscriber that waits for the prefetch to finish
        let got_0 = test.client.fetch(run_id, &data_0_key).unwrap();
        assert_eq!(got_0, data_0_val);
        let got_1 = test.client.fetch(run_id, &data_1_key).unwrap();
        assert_eq!(got_1, data_1_val);

        test.client.clean_up(run_id).unwrap();
    }

    fn setup() -> Test {
        let client_dir = tempfile::tempdir().unwrap();
        let server_addr: Url = "http://localhost:4000".try_into().unwrap();
        let client = RemoteStore::new(client_dir, 10 * 1024 * 1024, server_addr).unwrap();

        Test { client }
    }
}
