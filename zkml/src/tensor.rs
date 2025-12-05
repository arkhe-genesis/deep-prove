#![allow(clippy::needless_range_loop)]

mod burn_wrapper;

pub use burn_wrapper::{
    BShape, BTensorKind, Conversion, IntoBTensor, TensorTypeParam, WrappedModuleFn, WrappedTensor,
};

use crate::{
    NextPowerOfTwo, ScalingFactor, backend::Backend, layers::convolution, number::Number,
    shape::Shape, to_field,
};
use anyhow::{Result, bail, ensure};
use burn::tensor::{Int, Tensor as BTensor, TensorData};
use ceno_p3::{
    field::{Field, FieldAlgebra, TwoAdicField},
    goldilocks::Goldilocks,
};
use ff_ext::{ExtensionField, GoldilocksExt2};
use itertools::Itertools;
use multilinear_extensions::mle::{IntoMLE, MultilinearExtension};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    prelude::ParallelSlice,
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    cmp::{Ordering, PartialEq, min},
    fmt::{self, Debug},
    ops::{Deref, DerefMut, Range},
    sync::{Arc, MappedRwLockReadGuard, RwLock, RwLockReadGuard},
};
use tenstore::{GenStore, GenericStore, StorageKey, StoreError};

use crate::{
    Element, layers::pooling::MAXPOOL2D_KERNEL_SIZE, quantization::Fieldizer, to_bit_sequence_le,
};

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    derive_more::From,
    derive_more::Into,
    derive_more::Display,
)]
#[display("{_0}")]
pub struct CommitmentId(String);

impl From<&str> for CommitmentId {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

impl<T> From<&StorageKey<T>> for CommitmentId {
    fn from(value: &StorageKey<T>) -> Self {
        Self(value.id().to_string())
    }
}

impl<T> From<StorageKey<T>> for CommitmentId {
    fn from(value: StorageKey<T>) -> Self {
        Self(value.id().to_string())
    }
}

impl<T> From<CommitmentId> for StorageKey<T> {
    fn from(value: CommitmentId) -> Self {
        StorageKey::<T>::new(value.0)
    }
}

impl<T> From<&CommitmentId> for StorageKey<T> {
    fn from(value: &CommitmentId) -> Self {
        StorageKey::<T>::new(value.0.clone())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyedTensor<T> {
    pub(crate) key: StorageKey<T>,
    pub(crate) tensor: Tensor<T>,
}

impl<T> KeyedTensor<T> {
    pub fn new<S>(key: S, tensor: Tensor<T>) -> Self
    where
        S: Into<StorageKey<T>>,
    {
        Self {
            key: key.into(),
            tensor,
        }
    }

    pub fn storage_key(&self) -> &StorageKey<T> {
        &self.key
    }

    pub fn commitment_id(&self) -> CommitmentId {
        (&self.key).into()
    }

    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    pub fn tensor(&self) -> Tensor<T>
    where
        T: Clone,
    {
        self.tensor.clone()
    }

    pub fn map_tensor<U>(self, f: impl FnOnce(Tensor<T>) -> Tensor<U>) -> KeyedTensor<U> {
        self.try_map_tensor(|t| Ok(f(t))).unwrap()
    }

    pub fn try_map_tensor<U>(
        self,
        f: impl FnOnce(Tensor<T>) -> anyhow::Result<Tensor<U>>,
    ) -> anyhow::Result<KeyedTensor<U>> {
        Ok(KeyedTensor {
            key: self.key.cast::<U>(),
            tensor: f(self.tensor)?,
        })
    }

    pub fn new_map_tensor<U>(&self, f: impl FnOnce(&Tensor<T>) -> Tensor<U>) -> KeyedTensor<U> {
        KeyedTensor {
            key: self.key.cast::<U>(),
            tensor: f(&self.tensor),
        }
    }

    pub fn try_new_map_tensor<U>(
        &self,
        f: impl FnOnce(&Tensor<T>) -> anyhow::Result<Tensor<U>>,
    ) -> anyhow::Result<KeyedTensor<U>> {
        Ok(KeyedTensor {
            key: self.key.cast::<U>(),
            tensor: f(&self.tensor)?,
        })
    }
}

impl<T> Deref for KeyedTensor<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<T> DerefMut for KeyedTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

impl KeyedTensor<f32> {
    pub fn quantize(self, s: &ScalingFactor) -> KeyedTensor<Element> {
        let quantized_tensor = self.tensor.to_quantized(s);
        KeyedTensor {
            key: self.key.cast::<Element>(),
            tensor: quantized_tensor,
        }
    }
}

/// Chunk size tuned to allow for the compiler's auto vectorisation.
///
/// This value should be a multiple of the target's cache line (taking into
/// consideration the number of bytes, including padding, of the data being
/// iterated over). To support multiple architectures this should be the least
/// common multiple of all target architectures.
///
/// This value is also used to increase utilisation when utilising rayon,
/// reducing the overhead cost of the rayon framework for super cheap operation.
/// (like a simple addition or subtraction).
const AUTO_VECTORISATION_CHUNK: usize = 512;

/// Macro to generate code that can be auto-vectorised over two tensors.
///
/// The auto vectorisation works by:
///
/// - Using `as_chunks`, which guarantees the data will be split in a continuous
///   non-overlapping chunks.
/// - The above chunks are then converted to an array with sizes known at compile
///   time
/// - The data is then iterated over the above arrays using a simple incrementing
///   index
macro_rules! auto_vec_binop {
    ($self: ident, $other: ident, $op: tt) => {{
        assert!(
            $self.shape.product() == $other.shape.product(),
            "Shape mismatch for addition {:?} != {:?}",
            $self.shape,
            $other.shape,
        );

        let mut data = Vec::with_capacity($self.data.len());
        let spare_data = data.spare_capacity_mut();
        assert!(
            spare_data.len() >= $self.data.len(),
            "Preallocated vector must have enough capacity"
        );

        let (left_chunks, left_remainder) = $self.data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
        let (right_chunks, right_remainder) = $other.data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
        let (result_chunks, result_remainder) = spare_data.as_chunks_mut::<AUTO_VECTORISATION_CHUNK>();

        (left_chunks, right_chunks, result_chunks)
            .into_par_iter()
            .for_each(|(left, right, result)| {
                for i in 0..AUTO_VECTORISATION_CHUNK {
                    result[i].write(left[i] $op right[i]);
                }
            });

        // Handle remainder data
        for pos in 0..left_remainder.len()  {
            result_remainder[pos].write(left_remainder[pos] $op right_remainder[pos]);
        }

        // Safety: the memory was initialised above
        unsafe {
            data.set_len($self.data.len());
        }

        Tensor {
            shape: $self.shape.clone(),
            unpadded_shape: $self.unpadded_shape.clone(),
            data,
        }
    }}
}

/// Macro to apply an operation to each element of an array.
///
/// The expanded code is intended to be easily vectorisable, by using the
/// `as_chunks` api.
macro_rules! auto_vec_op {
    ($data: expr, $op: expr) => {{
        let mut result = Vec::with_capacity($data.len());
        let spare_result = result.spare_capacity_mut();
        assert!(
            spare_result.len() >= $data.len(),
            "Preallocated vector must have enough capacity"
        );

        let (left_chunks, left_remainder) = $data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
        let (result_chunks, result_remainder) =
            spare_result.as_chunks_mut::<AUTO_VECTORISATION_CHUNK>();

        (left_chunks, result_chunks)
            .into_par_iter()
            .for_each(|(left, result)| {
                for i in 0..AUTO_VECTORISATION_CHUNK {
                    result[i].write($op(left[i]));
                }
            });

        // Handle remainder data
        for pos in 0..left_remainder.len() {
            result_remainder[pos].write($op(left_remainder[pos]));
        }

        // Safety: the memory was initialised above
        unsafe {
            result.set_len($data.len());
        }

        result
    }};
}

/// Returns an n-th root of unity by starting with a 32nd root of unity and squaring it (32-n) times.
/// Each squaring operation halves the order of the root of unity:
///   - For n=16: squares it 16 times (32-16) to get a 16th root of unity
///   - For n=8:  squares it 24 times (32-8) to get an 8th root of unity
///   - For n=4:  squares it 28 times (32-4) to get a 4th root of unity
///
/// The initial ROOT_OF_UNITY constant is verified to be a 32nd root of unity in the field implementation.
pub fn get_root_of_unity<E: ExtensionField>(n: usize) -> E {
    let mut rou = E::from_bases(&[
        E::BaseField::two_adic_generator(Goldilocks::TWO_ADICITY),
        E::BaseField::ZERO,
    ]);

    for _ in 0..(32 - n) {
        rou = rou * rou;
    }

    rou
}

/// Returns a permutation to convert a vector from normal order to bit reverse
/// order.
///
/// This can be used to start a FFT using decimation in time or to finalise a
/// FFT with decimation in frequency.
pub fn bitreverse_permutation(length: usize) -> impl Iterator<Item = usize> {
    let shift = usize::BITS - length.ilog2();
    (0..length).map(move |i| i.reverse_bits() >> shift)
}

/// Applies a bitreverse order to the slice.
///
/// This can be used to start a FFT using decimation in time or to finalise a
/// FFT with decimation in frequency.
pub fn bitreverse<T>(d: &mut [T]) {
    for (orig, new) in bitreverse_permutation(d.len())
        .enumerate()
        // filter out duplicates
        .filter(|(orig, new)| orig < new)
    {
        d.swap(orig, new)
    }
}

/// Perform a radix-2 Cooley-Tukey FFT.
///
/// flag: false -> FFT
/// flag: true -> iFFT
pub fn fft<E: ExtensionField + Send + Sync>(v: &mut Vec<E>, flag: bool) -> Result<()> {
    ensure!(
        v.len().is_power_of_two(),
        "Input vector to fft must be a power of two",
    );

    let n = v.len();
    let logn = n.ilog2();

    // Perform bit reverse permutation. The code below performs decimation in
    // time (DIT), data is reordered prior to the butterflies.
    bitreverse(v);

    // Compute the twiddle factors
    let mut twiddle: Vec<E> = vec![E::ZERO; n];
    twiddle[0] = E::ONE;
    twiddle[1] = get_root_of_unity(logn as usize);

    if flag {
        twiddle[1] = twiddle[1].inverse();
    }
    for i in 2..n {
        twiddle[i] = twiddle[i - 1] * twiddle[1];
    }

    let mut i: usize = 2;
    while i <= n {
        v.par_chunks_mut(i).for_each(|chunk| {
            let half_i = i >> 1;
            for k in 0..half_i {
                let u = chunk[k];
                let l = chunk[k + half_i] * twiddle[n / i * k];
                chunk[k] = u + l;
                chunk[k + half_i] = u - l;
            }
        });
        i <<= 1;
    }

    if flag {
        let mut ilen = E::from_canonical_u64(n as u64);
        ilen = ilen.inverse();
        debug_assert_eq!(
            ilen * E::from_canonical_u64(n as u64),
            E::ONE,
            "Error in inv"
        );
        v.par_iter_mut().for_each(|val| {
            *val *= ilen;
        });
    }

    Ok(())
}

/// A handle to manage different representations of the same data.
///
/// Tensors are used in different contexts, each requiring a different
/// representation / implementation. This struct wraps the necessary metadata to
/// load, store, and transform the tensors to different representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub enum TensorHandle<T>
where
    T: TensorTypeParam,
{
    WrappedTensor {
        /// A unique key for this tensor.
        storage_key: StorageKey<Vec<T>>,

        /// Storage used to save or load the underlying data.
        #[serde(skip)]
        store: GenStore,

        /// Accelerated tensor data, if available.
        ///
        /// If the tensor data is not available, the handler will try to hydrate it
        /// by reading from the corresponding tensor, which might need to be read
        /// from the store.
        #[serde(skip)]
        wrapped_tensor: Arc<RwLock<Option<WrappedTensor<T>>>>,

        /// The shape of the tensor.
        shape: Shape,
        unpadded_shape: Shape,
    },
    Tensor {
        /// A unique key for this tensor.
        storage_key: StorageKey<Vec<T>>,

        /// Storage used to save or load the underlying data.
        #[serde(skip)]
        store: GenStore,

        /// Tensor data, if available.
        ///
        /// If the tensor data is not available, the handler will try to hydrate it
        /// by reading from the corresponding store.
        #[serde(skip)]
        tensor: Arc<RwLock<Option<Tensor<T>>>>,

        /// The shape of the tensor.
        shape: Shape,
        unpadded_shape: Shape,
    },
}

impl<T> TensorHandle<T>
where
    T: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
{
    /// Creates a [TensorHandle] from a [Tensor].
    pub(crate) fn from_tensor(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        tensor: Tensor<T>,
    ) -> Self {
        let shape = tensor.shape().clone();
        let unpadded_shape = tensor.unpadded_shape().clone();
        Self::Tensor {
            storage_key,
            store,
            tensor: Arc::new(RwLock::new(Some(tensor))),
            shape,
            unpadded_shape,
        }
    }

    /// Ensures this is a [WrappedTensor] variant.
    pub(crate) fn into_wrapped_tensor(self) -> anyhow::Result<Self> {
        match self {
            result @ TensorHandle::WrappedTensor { .. } => Ok(result),
            TensorHandle::Tensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            } => {
                let guard = tensor.read().expect("Lock should not be poisioned");
                let wrapped_tensor = match guard.deref() {
                    Some(tensor) => Some(tensor.try_into()?),
                    None => None,
                };
                Ok(TensorHandle::WrappedTensor {
                    storage_key,
                    store,
                    wrapped_tensor: Arc::new(RwLock::new(wrapped_tensor)),
                    shape,
                    unpadded_shape,
                })
            }
        }
    }

    /// Ensures this is a [Tensor] variant.
    pub(crate) fn into_dry_tensor(self) -> anyhow::Result<Self> {
        match self {
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => Ok(TensorHandle::Tensor {
                storage_key,
                store,
                tensor: Arc::new(RwLock::new(None)),
                shape,
                unpadded_shape,
            }),
            result @ TensorHandle::Tensor { .. } => Ok(result),
        }
    }

    /// Returns the [StorageKey] used to identify the data in the store.
    pub(crate) fn storage_key(&self) -> &StorageKey<Vec<T>> {
        match self {
            TensorHandle::WrappedTensor {
                ref storage_key, ..
            } => storage_key,
            TensorHandle::Tensor {
                ref storage_key, ..
            } => storage_key,
        }
    }

    /// Returns a reference to the shape of this tensor.
    pub(crate) fn shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor { ref shape, .. } => shape,
            TensorHandle::Tensor { ref shape, .. } => shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn unpadded_shape(&self) -> &Shape {
        match self {
            TensorHandle::WrappedTensor {
                ref unpadded_shape, ..
            } => unpadded_shape,
            TensorHandle::Tensor {
                ref unpadded_shape, ..
            } => unpadded_shape,
        }
    }

    /// Returns a reference to the unpadded shape of this tensor.
    pub(crate) fn store(&self) -> &GenStore {
        match self {
            TensorHandle::WrappedTensor { ref store, .. } => store,
            TensorHandle::Tensor { ref store, .. } => store,
        }
    }

    /// Sets the `store` for this handle.
    pub(crate) fn attach_store(&mut self, value: GenStore) {
        match self {
            TensorHandle::WrappedTensor { ref mut store, .. } => *store = value,
            TensorHandle::Tensor { ref mut store, .. } => *store = value,
        }
    }

    /// Utility to load data from the store.
    fn load(&self) -> anyhow::Result<()> {
        match self {
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                wrapped_tensor,
                shape,
                ..
            } => {
                let mut guard = wrapped_tensor.write().expect("Lock should not be poisoned");

                if guard.is_none() {
                    let tensor = store.fetch(storage_key).map(|data| {
                        let data = TensorData::new(data, BShape::from(shape.clone()));
                        WrappedTensor::from_data(data)
                    })??;
                    *guard = Some(tensor);
                }
            }
            TensorHandle::Tensor {
                storage_key,
                store,
                tensor,
                shape,
                unpadded_shape,
            } => {
                let mut guard = tensor.write().expect("Lock should not be poisoned");

                if guard.is_none() {
                    let tensor = store.fetch(storage_key).map(|data| {
                        Tensor::new_with_unpadded_shape(shape.clone(), unpadded_shape.clone(), data)
                    })??;
                    *guard = Some(tensor);
                }
            }
        }

        Ok(())
    }
    /// Returns a reference to the cached [`Tensor`].
    ///
    /// NOTE: If the [`Tensor`] is not cached, this will load the data from
    /// the store.
    pub fn tensor(&self) -> Result<MappedRwLockReadGuard<'_, Tensor<T>>> {
        match self {
            TensorHandle::WrappedTensor { .. } => {
                bail!("Tensor is unavaiable for a wrapped tensor handler")
            }
            TensorHandle::Tensor { tensor, .. } => loop {
                {
                    let guard = tensor.read().expect("Lock should not be poisoned");
                    if guard.is_some() {
                        let res = RwLockReadGuard::map(guard, |v| match v {
                            Some(ref v) => v,
                            None => {
                                unreachable!(
                                    "The option was checked above, this is in a read only region"
                                )
                            }
                        });
                        return Ok(res);
                    }
                }
                self.load()?;
            },
        }
    }

    /// Returns a reference to the cached [`WrappedTensor`].
    ///
    /// NOTE: If the [`WrappedTensor`] is not cached, it will be created, to create
    /// the tensor the corresponding [`Tensor`] must be available, if it is not, the
    /// data will be loaded from the store.
    pub fn wrapped_tensor(&self) -> Result<MappedRwLockReadGuard<'_, WrappedTensor<T>>> {
        match self {
            TensorHandle::WrappedTensor { wrapped_tensor, .. } => loop {
                {
                    let guard = wrapped_tensor.read().expect("Lock should not be poisoned");
                    if guard.is_some() {
                        let res = RwLockReadGuard::map(guard, |v| match v {
                            Some(ref v) => v,
                            None => {
                                unreachable!(
                                    "The option was checked above, this is in a read only region"
                                )
                            }
                        });
                        return Ok(res);
                    }
                    self.load()?;
                }
            },
            TensorHandle::Tensor { .. } => {
                bail!("Wrapped tensor is unavaiable for a tensor handler")
            }
        }
    }

    /// Dries the current handle.
    ///
    /// Drying a handle frees the cached values to free memory.
    pub(crate) fn dry(&self) {
        match self {
            TensorHandle::WrappedTensor { wrapped_tensor, .. } => {
                let mut guard = wrapped_tensor
                    .write()
                    .expect("Lock should not be poisioned");
                *guard = None;
            }
            TensorHandle::Tensor { tensor, .. } => {
                let mut guard = tensor.write().expect("Lock should not be poisioned");
                *guard = None;
            }
        }
    }

    /// Ensure the transformed version of this tensor with type [S] exists in the store.
    ///
    /// If the transformed version does not exist yet, apply `f` over `self` and
    /// save it to the store.
    pub(crate) fn cast<S, F>(&self, f: F) -> Result<TensorHandle<S>, StoreError>
    where
        S: TensorTypeParam + Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> S,
    {
        match self {
            TensorHandle::WrappedTensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::WrappedTensor {
                    storage_key,
                    store: store.clone(),
                    wrapped_tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
            TensorHandle::Tensor {
                storage_key,
                store,
                shape,
                unpadded_shape,
                ..
            } => {
                let storage_key =
                    store.cast(storage_key, |xs| xs.iter().map(&f).collect::<Vec<S>>())?;
                Ok(TensorHandle::<S>::Tensor {
                    storage_key,
                    store: store.clone(),
                    tensor: Default::default(),
                    shape: shape.clone(),
                    unpadded_shape: unpadded_shape.clone(),
                })
            }
        }
    }

    /// Loads the transformed version of this tensor with type [S].
    ///
    /// If the transformed version does not yet exist, apply `f` over `self`,
    /// save it to store and returns a copy.
    pub(crate) fn hydrated_cast<S, F>(&self, f: F) -> Result<Tensor<S>>
    where
        S: Serialize + for<'a> Deserialize<'a>,
        F: Fn(&T) -> S,
    {
        let result = self
            .store()
            .cast_and_fetch(self.storage_key(), |xs| {
                xs.iter().map(&f).collect::<Vec<S>>()
            })
            .map(|bytes| {
                Tensor::new_with_unpadded_shape(
                    self.shape().clone(),
                    self.unpadded_shape().clone(),
                    bytes.1,
                )
            })?;

        result
    }
}

impl<T> TensorHandle<T>
where
    T: Serialize + for<'a> Deserialize<'a> + TensorTypeParam,
{
    pub(crate) fn from_wrapped_tensor(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        wrapped_tensor: WrappedTensor<T>,
    ) -> Self {
        let shape = Shape::from(wrapped_tensor.shape());
        let unpadded_shape = Shape::from(wrapped_tensor.unpadded_shape());
        Self::WrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        }
    }

    pub(crate) fn from_wrapped_tensor_with_unpadded_shape(
        storage_key: StorageKey<Vec<T>>,
        store: GenStore,
        mut wrapped_tensor: WrappedTensor<T>,
        unpadded_shape: Shape,
    ) -> Self {
        let shape = Shape::from(wrapped_tensor.shape());
        wrapped_tensor.set_unpadded_shape(unpadded_shape.clone().into());
        Self::WrappedTensor {
            storage_key,
            store,
            wrapped_tensor: Arc::new(RwLock::new(Some(wrapped_tensor))),
            shape,
            unpadded_shape,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::Index, derive_more::IndexMut)]
pub struct Tensor<T> {
    // Indexing the `Tensor` indexes the underlying storage.
    #[index]
    #[index_mut]
    data: Vec<T>,
    shape: Shape,
    unpadded_shape: Shape,
}
impl<T> Tensor<T> {
    /// Create a new tensor with given shape and data
    pub fn new(shape: Shape, data: Vec<T>) -> Result<Self> {
        Self::new_with_unpadded_shape(shape.clone(), shape, data)
    }

    /// Create a new tensor with given shape, unpadded shape and data
    pub fn new_with_unpadded_shape(
        shape: Shape,
        unpadded_shape: Shape,
        data: Vec<T>,
    ) -> Result<Self> {
        ensure!(
            shape.product() == data.len(),
            "Shape does not match data length: shape {:?}->{} vs data.len() {}",
            shape,
            shape.product(),
            data.len(),
        );
        Ok(Self {
            data,
            shape,
            unpadded_shape,
        })
    }

    /// Create a new tensor with the given shapes & data, not ensuring that they
    /// actually match.
    pub fn new_unchecked(shape: Shape, data: Vec<T>) -> Self {
        Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }

    /// Return an immutable reference to this tensor data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Return an immutable reference to this tensor data.
    pub fn data_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// Consume this tensore, returning its backing.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Return the number of elements contained in this tensor, independently of
    /// its shape.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    pub fn num_vars(&self) -> usize {
        self.shape.num_vars().iter().sum()
    }

    /// Iterates over the data in the tensor
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Mutable Iterator over the data in the tensor
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Returns an iterator that yields slices of the last dimension.
    ///
    /// For a tensor of shape `[2, 3, 3]`, it will yield 6 slices `2 * 3` of 3 elements each.
    pub fn slice_last_dim(&self) -> impl Iterator<Item = &[T]> {
        let (it, _) = self.slice_on_dim(self.shape.len() - 2);
        it
    }

    /// Consumes this tensor and returns a new one with the given dimensions flattened.
    ///
    /// # Panics
    ///
    /// If the given dimensions are out-of-bounds.
    pub fn flatten(mut self, dims: Range<usize>) -> Self {
        self.shape = self.shape.flatten(dims);
        self
    }

    /// Returns a zero-copy logical flatten of all leading dimensions into rows while preserving the last dimension.
    /// For shape [d0,d1,...,dk-1, last] this yields (data slice, rows = d0*..*dk-1, last_dim = last).
    pub fn flatten_leading_dims_view(&self) -> (&[T], usize, usize) {
        let rank = self.shape.len();
        assert!(rank >= 2, "Need rank >=2 to flatten leading dims view");
        let last_dim = self.shape.dim(rank - 1);
        let rows: usize = if rank == 2 {
            self.shape.dim(0)
        } else {
            (0..rank - 1).map(|d| self.shape.dim(d)).product()
        };
        (&self.data, rows, last_dim)
    }

    /// Returns an iterator of slices whose length corresponds to the subspace
    /// the dimension represents. Note dim is the dimension _index_ (0-based indexing).
    /// Example: if dimension is [2,3,4], and we call `slice_on_dim(1)`,
    /// it will yield 2x3 slices of 4 elements each. If we call `slice_on_dim(0)`,
    /// it will yield 2 slices of 3x4=12 element each.
    /// If dim is the last dimension, it will simply yield a slice of the whole tensor.
    /// The shape returned is the shape of each slice. The shape is the same as the shape of the tensor
    /// if the dim is the last dimension or more
    pub fn slice_on_dim(&self, dim: usize) -> (impl Iterator<Item = &[T]>, Shape) {
        assert!(
            dim < self.shape.len(),
            "can't slice on dim {:?} if shape is {:?}",
            dim,
            self.shape
        );
        let (stride, shape) = if dim < self.shape.len() - 1 {
            let slice = self.shape.slice(dim + 1..);
            (slice.product(), slice)
        } else {
            (self.shape.product(), self.shape.clone())
        };
        (self.data.chunks(stride), shape)
    }

    // Concatenate the other tensor to the first one.
    // RESTRICTIOn: self shape is [a1,a2...,an] we
    // expect other shape to be [a2...,an] OR [1, a2...,an]
    // The new shape of self will be [a1+1,...an]
    // In other words, we only concatenate another vector if it's exactly size of the highest dimension
    // If it's 2d, then we expect other to be a vector
    pub fn concat(&mut self, other: Self) -> Result<()> {
        // make sure that the all dimension but the highest one are the same
        let common_shape = self.shape.len().min(other.shape.len());
        let added_higher = if common_shape < self.shape.len() {
            ensure!(
                self.shape
                    .iter()
                    .rev()
                    .zip(other.shape.iter().rev())
                    .take(common_shape)
                    .all(|(a, b)| a == b)
            );
            ensure!(common_shape + 1 == self.shape.len());
            1
        } else {
            ensure!(common_shape == self.shape.len());
            *other.shape.first().unwrap()
        };
        // then the new shape has this higher dimension + 1 simply
        // common_shape since 0-based indexing
        *self.shape.get_mut(0).unwrap() += added_higher;
        self.data.extend(other.data);

        Ok(())
    }

    /// Adds a new dimension to the tensor with size 1.
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than the shape size.
    pub fn unsqueeze(self, index: usize) -> Self {
        let new_shape = self.shape.insert(index, 1);
        Self {
            data: self.data,
            shape: new_shape,
            unpadded_shape: self.unpadded_shape,
        }
    }

    /// Removes a dimension from the tensor.
    ///
    /// # Panics
    ///
    /// Panics if `index` is larger than the shape size, or if the
    /// removed dimension size is not 1.
    pub fn squeeze(mut self, index: usize) -> Result<Self> {
        ensure!(
            self.shape[index] == 1,
            "The dimension to be squeezed must have a value of 1"
        );
        self.shape.remove(index);
        Ok(self)
    }

    /// Get the number of rows from the matrix
    pub fn nrows_2d(&self) -> Result<usize> {
        let mut cols = 0;
        let dims = self.shape();
        if self.shape.is_matrix() {
            cols = dims[0];
        } else if self.shape.is_convolution() {
            cols = dims[0] * dims[2] * dims[2];
        }
        ensure!(cols != 0, "Tensor is not a matrix or convolution");
        Ok(cols)
    }

    /// Get the number of cols from the matrix
    pub fn ncols_2d(&self) -> Result<usize> {
        let mut cols = 0;
        let dims = self.shape();
        if self.shape.is_matrix() {
            cols = dims[1];
        } else if self.shape.is_convolution() {
            cols = dims[1] * dims[2] * dims[2];
        }
        ensure!(cols != 0, "Tensor is not a matrix or convolution");
        // assert!(self.is_matrix(), "Tensor is not a matrix");
        // let dims = self.dims();

        Ok(cols)
    }

    /// Get the dimensions of the tensor
    pub fn shape(&self) -> &Shape {
        assert!(!self.shape.is_empty(), "Empty tensor");
        &self.shape
    }

    pub fn unpadded_shape(&self) -> &Shape {
        &self.unpadded_shape
    }

    /// Get the dimensions of the tensor
    pub fn shape_mut(&mut self) -> &mut Shape {
        &mut self.shape
    }

    /// Returns the number of dimensions the [`Tensor`] has
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns a reference to the tensor data.
    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    /// Consumes the tensor and return its data.
    pub fn get_data_into(self) -> Vec<T> {
        self.data
    }

    /// Returns the size of the given dimension.
    ///
    /// # Panics
    ///
    /// If the shape rank is lower-or-equal to `dim`.
    pub fn dim(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    /// Returns the tensor data converted to extension field elements.
    pub fn to_field<F: ExtensionField>(&self) -> Vec<F>
    where
        T: Fieldizer<F>,
    {
        to_field::<T, F, _>(&self.data)
    }
}

impl<T> AsRef<Tensor<T>> for Tensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

impl Tensor<Element> {
    /// Returns the maximum size in bits possible if this tensor is treated as a matrix inside
    /// a matrix vector/matrix multiplication. It requires the optional inputs to specify the range
    // of the quantized values in `self` and in the other matrix being multiplied with `self`
    pub fn matmul_output_bitsize(
        &self,
        quantized_self_input_range: Option<usize>,
        quantized_other_input_range: Option<usize>,
    ) -> Result<usize> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        Ok(self
            .shape
            .matmul_output_bitsize(quantized_self_input_range, quantized_other_input_range))
    }

    pub fn dequantize(&self, s: &ScalingFactor) -> Tensor<f32> {
        let data = self
            .data
            .iter()
            .map(|e| s.dequantize(e))
            .collect::<Vec<_>>();
        Tensor {
            shape: self.shape.clone(),
            data,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    /// Converts this [Tensor<Element>] into [MultilinearExtension].
    ///
    /// This will convert the element into an extension field and convert that
    /// into a multilinear extension.
    ///
    /// see [Tensor::into_mle_2d].
    ///
    /// # Panics
    ///
    /// If the input is not a 2D tensor or if either dimension is not a power of two.
    pub fn to_2d_mle<F: ExtensionField>(&self) -> Result<MultilinearExtension<'static, F>> {
        Tensor::<F>::from(self).into_mle_2d()
    }

    /// Converts this [Tensor<Element>] into a [MultilinearExtension].
    ///
    /// This will convert the element into an extension field and convert that
    /// into a multilinear extension.
    ///
    /// This method does not enforce a dimensionality.
    ///
    /// see [Tensor::into_mle].
    ///
    /// # Panics
    ///
    /// If the number of elements in the Tensor is not a power of two.
    pub fn to_field_mle<F: ExtensionField>(&self) -> MultilinearExtension<'static, F> {
        Tensor::<F>::from(self).into_mle()
    }

    /// Consumes this tensor and creates a [burn::tensor::Tensor].
    pub fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D, Int> {
        IntoBTensor::to_btensor(self)
    }
}

impl<F: ExtensionField> Tensor<F> {
    /// Clone this [Tensor] and convert into a [MultilinearExtension].
    ///
    /// see [Tensor::into_mle_2d].
    pub fn to_mle_2d(&self) -> Result<MultilinearExtension<'static, F>> {
        self.clone().into_mle_2d()
    }

    /// Consumes this [Tensor] into a [MultilinearExtension].
    ///
    /// The [Tensor] must be in evaluation form.
    ///
    /// # Panics
    ///
    /// - If the tensor is not 2D.
    /// - If either dimension is not a power-of-two.
    pub fn into_mle_2d(self) -> Result<MultilinearExtension<'static, F>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        ensure!(
            self.nrows_2d()?.is_power_of_two(),
            "number of rows {} is not a power of two",
            self.nrows_2d()?
        );
        ensure!(
            self.ncols_2d()?.is_power_of_two(),
            "number of columns {} is not a power of two",
            self.ncols_2d()?
        );
        // N variable to address 2^N rows and M variables to address 2^M columns
        let num_vars = self.nrows_2d()?.ilog2() + self.ncols_2d()?.ilog2();
        Ok(MultilinearExtension::from_evaluations_ext_vec(
            num_vars as usize,
            self.data,
        ))
    }
}

impl<F: Field> Tensor<F> {
    /// Clone this [Tensor] and convert into a [MultilinearExtension].
    ///
    /// see [Tensor::into_mle].
    pub fn to_mle<E: ExtensionField>(&self) -> MultilinearExtension<'_, E> {
        self.data.clone().into_mle()
    }

    /// Consumes this [Tensor] into a [MultilinearExtension].
    ///
    /// The [Tensor] must be in evaluation form.
    ///
    /// # Panics
    ///
    /// - If the number of elements in the tensor is not a power of two.
    pub fn into_mle<E: ExtensionField>(self) -> MultilinearExtension<'static, E> {
        self.data.into_mle()
    }
}

impl<F: ExtensionField> From<&Tensor<Element>> for Tensor<F> {
    fn from(value: &Tensor<Element>) -> Self {
        Self {
            data: value.to_field::<F>(),
            shape: value.shape.clone(),
            unpadded_shape: value.unpadded_shape.clone(),
        }
    }
}

impl Tensor<f32> {
    pub fn to_quantized(&self, s: &ScalingFactor) -> Tensor<Element> {
        let data = self.data.iter().map(|x| s.quantize(x)).collect::<Vec<_>>();
        Tensor {
            shape: self.shape.clone(),
            data,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    /// Consumes this tensor and creates a [burn::tensor::Tensor].
    pub fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D> {
        IntoBTensor::to_btensor(self)
    }
}

impl<T: Clone> Tensor<T> {
    pub fn to_flatten(&self) -> Self {
        let new_data = self.get_data().to_vec();
        let new_shape = vec![new_data.len()];
        Self {
            data: new_data,
            shape: new_shape.into(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
    pub fn matrix_from_coeffs(data: Vec<Vec<T>>) -> anyhow::Result<Self> {
        let n_rows = data.len();
        let n_cols = data.first().expect("at least one row in a matrix").len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();
        if data.len() != n_rows * n_cols {
            bail!(
                "Number of rows and columns do not match with the total number of values in the Vec<Vec<>>"
            );
        };
        let shape: Shape = vec![n_rows, n_cols].into();

        Ok(Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape,
        })
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn row_to_boolean_2d<F: ExtensionField>(
        &self,
        row: usize,
    ) -> Result<impl Iterator<Item = F>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        let (nvars_rows, _) = self.shape().num_vars_2d();
        Ok(to_bit_sequence_le(row, nvars_rows).map(|b| F::from_canonical_u64(b as u64)))
    }
    /// Returns the boolean iterator indicating the given row in the right endianness to be
    /// evaluated by an MLE
    pub fn col_to_boolean_2d<F: ExtensionField>(
        &self,
        col: usize,
    ) -> Result<impl Iterator<Item = F>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        let (_, nvars_col) = self.shape().num_vars_2d();
        Ok(to_bit_sequence_le(col, nvars_col).map(|b| F::from_canonical_u64(b as u64)))
    }
    /// From a given row and a given column, return the vector of field elements in the right
    /// format to evaluate the MLE.
    /// little endian so we need to read cols before rows
    pub fn position_to_boolean_2d<F: ExtensionField>(
        &self,
        row: usize,
        col: usize,
    ) -> Result<Vec<F>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        Ok(self
            .col_to_boolean_2d(col)?
            .chain(self.row_to_boolean_2d(row)?)
            .collect_vec())
    }

    /// Reshape the tensor, ensuring that the new shape is compatible with the
    /// tensor cardinality.
    pub(crate) fn reshape(&mut self, new_shape: Shape) -> Result<()> {
        ensure!(
            self.shape.product() == new_shape.product(),
            "Shape mismatch for reshape: current {:?} ({}), required {:?} ({})",
            self.shape,
            self.shape.product(),
            new_shape,
            new_shape.product()
        );
        // If the tensor has not yet been padded also update the unpadded shape
        if self.shape == self.unpadded_shape {
            self.unpadded_shape = new_shape.clone();
        }
        self.shape = new_shape;

        Ok(())
    }

    /// Force-set the tensor shape, not checking that the new shape is
    /// compatible with the tensor cardinality.
    pub fn reshape_unchecked(&mut self, new_shape: Shape) {
        self.shape = new_shape;
    }

    /// Chainable version of [`reshape`]
    pub fn reshaped(mut self, new_shape: Shape) -> Result<Tensor<T>> {
        self.reshape(new_shape)?;
        Ok(self)
    }
}

struct ArgMax<T> {
    /// The biggest value in the tensor.
    value: T,

    /// The position of the biggest value.
    position: usize,
}

impl<T: Number> Tensor<T> {
    /// Finds the maximum value in the tensor and returns its value and position.
    fn find_maximum(&self) -> ArgMax<T> {
        let (position, value) = self.data.par_iter().cloned().enumerate().reduce(
            || (usize::MAX, T::MIN),
            |acc, (position, value)| match acc.1.compare(&value) {
                Ordering::Less => (position, value),
                _ => acc,
            },
        );

        ArgMax { value, position }
    }

    /// Instantiate a new tensor with `shape` initialised to `default`.
    pub fn zeros(shape: Shape) -> Self {
        Self::initialised(shape, T::zero())
    }

    /// Creates a new [Tensor] with `shape` initialised to `T::unit`.
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let shape = Shape::new(vec![2, 2]);
    /// let tensor = Tensor::<Element>::one(shape);
    /// assert_eq!(tensor.data(), [1, 1, 1, 1]);
    /// ```
    pub fn one(shape: Shape) -> Self {
        Tensor {
            data: vec![T::unit(); shape.numel()],
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }

    /// Returns the first position of the largest element in the [Tensor].
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let tensor = Tensor::<Element>::new(Shape::new(vec![4, 2]), vec![3, 1, 0, 11, 7, 11, 9, 2]).unwrap();
    /// assert_eq!(tensor.argmax(), 3);
    /// ```
    pub fn argmax(&self) -> usize {
        self.find_maximum().position
    }

    /// Returns the largest value in the [Tensor].
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let tensor = Tensor::<Element>::new(Shape::new(vec![4, 2]), vec![3, 1, 0, 11, 7, 11, 9, 2]).unwrap();
    /// assert_eq!(tensor.max_value(), 11);
    /// ```
    pub fn max_value(&self) -> T {
        self.find_maximum().value
    }

    /// Returns the the largest absolute element in the [Tensor].
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let tensor = Tensor::<Element>::new(Shape::new(vec![7]), vec![3, 1, 0, -11, 7, 9,
    /// 2]).unwrap();
    /// assert_eq!(tensor.max_abs_output(), 11);
    /// ```
    pub fn max_abs_output(&self) -> T {
        self.data
            .par_iter()
            .cloned()
            .reduce(|| T::zero(), |max, x| max.cmp_max(&x.absolute_value()))
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        auto_vec_binop!(self, other, +)
    }

    /// Add a vector to each sub-tensor of the second dimension of the tensor
    /// If self is 2d, then add a vector to each row of self.
    pub fn add_dim2(&self, other: &Tensor<T>) -> Tensor<T> {
        assert!(self.shape.len() == 2, "Tensor is not a matrix");
        assert!(other.shape.len() == 1, "Tensor is not a vector");
        assert!(
            self.shape[1] == other.shape[0],
            "Shape mismatch for addition2: {:?} != {:?}",
            self.shape,
            other.shape
        );
        let data = self
            .data
            .par_chunks(self.shape[1])
            .flat_map_iter(|chunk| chunk.iter().zip(other.data.iter()).map(|(a, b)| *a + *b))
            .collect::<Vec<_>>();
        Tensor {
            shape: self.shape.clone(),
            data,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
        auto_vec_binop!(self, other, -)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        auto_vec_binop!(self, other, *)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &T) -> Tensor<T> {
        let scalar = *scalar;

        let data = auto_vec_op!(self.data, |el| el * scalar);

        Tensor {
            data,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    /// Scalar multiplication with f32.
    pub fn scalar_mul_f32<N2: Number>(&self, scalar: N2) -> Tensor<T> {
        let scaled = self
            .data
            .par_iter()
            .map(|x| T::from_f32(x.to_f32()? * scalar.to_f32()?))
            .collect::<anyhow::Result<Vec<_>>>()
            .expect("Failed to scale tensor");
        Tensor {
            data: scaled,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    /// Flattens the tensor into a 1D.
    pub fn to_1d(&mut self) {
        self.shape = Shape::new(vec![self.shape.product()]);
    }

    /// Perform matrix-matrix multiplication
    pub fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        ensure!(
            self.shape.is_matrix() && other.shape.is_matrix(),
            "Both tensors must be 2D for matrix multiplication."
        );
        let (m, n) = (self.shape[0], self.shape[1]);
        let (n2, p) = (other.shape[0], other.shape[1]);
        ensure!(
            n == n2,
            "Matrix multiplication shape mismatch: {:?} cannot be multiplied with {:?}",
            self.shape,
            other.shape
        );

        let mut result = Tensor::zeros(vec![m, p].into());

        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, res)| {
                let i = index / p;
                let j = index % p;

                *res = (0..n)
                    .map(|k| self.data[i * n + k] * other.data[k * p + j])
                    .sum::<T>();
            });

        Ok(result)
    }
    /// Perform matrix-vector multiplication
    /// TODO: actually getting the result should be done via proper tensor-like libraries
    pub fn matvec(&self, vector: &Tensor<T>) -> Result<Tensor<T>> {
        ensure!(self.shape.is_matrix(), "First argument must be a matrix.");
        ensure!(
            vector.shape.is_vector(),
            "Second argument must be a vector."
        );

        let (m, n) = (self.shape[0], self.shape[1]);
        let vec_len = vector.shape[0];

        ensure!(n == vec_len, "Matrix columns must match vector size.");

        let mut result = Tensor::zeros(vec![m].into());

        result.data.par_iter_mut().enumerate().for_each(|(i, res)| {
            *res = (0..n)
                .map(|j| self.data[i * n + j] * vector.data[j])
                .sum::<T>();
        });

        Ok(result)
    }
    /// Transpose the matrix (2D tensor)
    pub fn transpose(&self) -> Result<Tensor<T>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix.");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut result = Tensor::zeros(vec![n, m].into());
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                let i = idx % m; // Row in the result matrix
                let j = idx / m; // Column in the result matrix
                *val = self.data[i * n + j];
            });

        Ok(result)
    }
    /// Concatenate a matrix (2D tensor) with a vector (1D tensor) as columns
    pub fn concat_matvec_col(&self, vector: &Tensor<T>) -> Result<Tensor<T>> {
        ensure!(self.shape.is_matrix(), "First tensor is not a matrix.");
        ensure!(vector.shape.is_vector(), "Second tensor is not a vector.");

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let vector_len = vector.shape[0];

        ensure!(
            rows == vector_len,
            "Matrix row count must match vector length."
        );

        let new_cols = cols + 1;
        let mut result = Tensor::zeros(vec![rows, new_cols].into());

        result
            .data
            .par_chunks_mut(new_cols)
            .enumerate()
            .for_each(|(i, row)| {
                row[..cols].copy_from_slice(&self.data[i * cols..(i + 1) * cols]); // Copy matrix row
                row[cols] = vector.data[i]; // Append vector element as the last column
            });

        Ok(result)
    }
    /// Reshapes the matrix to have at least the specified dimensions while preserving all data.
    pub fn reshape_to_fit_inplace_2d(&mut self, new_shape: Shape) -> Result<()> {
        let old_rows = self.nrows_2d()?;
        let old_cols = self.ncols_2d()?;

        ensure!(new_shape.len() == 2, "Tensor is not matrix");
        let new_rows = new_shape[0];
        let new_cols = new_shape[1];
        // Ensure we never lose information by requiring the new dimensions to be at least
        // as large as the original ones
        ensure!(
            new_rows >= old_rows,
            "Cannot shrink matrix rows from {old_rows} to {new_rows} - would lose information"
        );
        ensure!(
            new_cols >= old_cols,
            "Cannot shrink matrix columns from {old_cols} to {new_cols} - would lose information"
        );

        let new_data: Vec<T> = (0..new_rows * new_cols)
            .into_par_iter()
            .map(|idx| {
                let i = idx / new_cols;
                let j = idx % new_cols;
                if i < old_rows && j < old_cols {
                    self.data[i * old_cols + j]
                } else {
                    T::default() // Zero or default for padding
                }
            })
            .collect();

        self.shape = new_shape;
        self.data = new_data;

        Ok(())
    }
    pub fn maxpool2d(&self, kernel_size: usize, stride: usize) -> Result<Tensor<T>> {
        ensure!(
            kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {MAXPOOL2D_KERNEL_SIZE}"
        );
        ensure!(
            stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {MAXPOOL2D_KERNEL_SIZE}"
        );

        let dims = self.rank();
        ensure!(dims >= 2, "Input tensor must have at least 2 dimensions.");

        let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        // Assumes dilation = 1
        ensure!(
            h >= kernel_size,
            "Kernel size ({kernel_size}) is larger than input dimensions ({h}, {w})"
        );
        let out_h = (h - kernel_size) / stride + 1;
        let out_w = (w - kernel_size) / stride + 1;

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let output: Vec<T> = (0..outer_dims * out_h * out_w)
            .into_par_iter()
            .map(|flat_idx| {
                let n = flat_idx / (out_h * out_w);
                let i = (flat_idx / out_w) % out_h;
                let j = flat_idx % out_w;

                let matrix_idx = n * (h * w);
                let src_idx = matrix_idx + (i * stride) * w + (j * stride);
                let mut max_val = self.data[src_idx];

                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let src_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                        let value = self.data[src_idx];
                        max_val = max_val.cmp_max(&value);
                    }
                }

                max_val
            })
            .collect();

        let mut new_shape = self.shape.clone();
        new_shape[dims - 2] = out_h;
        new_shape[dims - 1] = out_w;

        Ok(Tensor {
            data: output,
            shape: new_shape,
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }

    // Replaces every value of a tensor with the maxpool of its kernel
    pub fn padded_maxpool2d(&self) -> Result<(Tensor<T>, Tensor<T>)> {
        let kernel_size = MAXPOOL2D_KERNEL_SIZE;
        let stride = MAXPOOL2D_KERNEL_SIZE;

        let maxpool_result = self.maxpool2d(kernel_size, stride)?;

        let dims: usize = self.rank();
        ensure!(dims >= 2, "Input tensor must have at least 2 dimensions.");

        let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

        ensure!(
            h % MAXPOOL2D_KERNEL_SIZE == 0,
            "Currently works only with kernel size {MAXPOOL2D_KERNEL_SIZE}"
        );
        ensure!(
            w % MAXPOOL2D_KERNEL_SIZE == 0,
            "Currently works only with stride size {MAXPOOL2D_KERNEL_SIZE}"
        );

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let maxpool_h = (h - kernel_size) / stride + 1;
        let maxpool_w = (w - kernel_size) / stride + 1;

        let padded_maxpool_data: Vec<T> = (0..outer_dims * h * w)
            .into_par_iter()
            .map(|out_idx| {
                let n = out_idx / (h * w);
                let i_full = (out_idx / w) % h;
                let j_full = out_idx % w;

                let i = i_full / stride;
                let j = j_full / stride;

                let maxpool_idx = n * maxpool_h * maxpool_w + i * maxpool_w + j;
                maxpool_result.data[maxpool_idx]
            })
            .collect();

        let padded_maxpool_tensor = Tensor {
            data: padded_maxpool_data,
            shape: self.shape().clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        };

        Ok((self.maxpool2d(kernel_size, stride)?, padded_maxpool_tensor))
    }

    // Applies a 2-dimensional convolution.
    pub fn conv2d(
        &self,
        kernels: &Tensor<T>,
        bias: &Tensor<T>,
        stride: usize,
    ) -> Result<Tensor<T>> {
        convolution::conv2d(self, kernels, bias, stride)
    }

    pub fn to_f32(&self) -> anyhow::Result<Tensor<f32>> {
        Ok(Tensor {
            data: self
                .data
                .iter()
                .map(Number::to_f32)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }
    /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
    /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular matrix must be square)
    /// - `num_matrices` specifies how many matrices to make
    /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `1`s on the first row instead of 1, and offset of `-1` means all `zeroes`
    pub fn tril(matrix_dim: usize, num_matrices: usize, diag: i32) -> Result<Tensor<T>> {
        Self::tri(matrix_dim, num_matrices, diag, T::unit(), T::default())
    }

    /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
    /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular matrix must be square)
    /// - `num_matrices` specifies how many matrices to make
    /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `lower_val`s on the first row instead of 1, and offset of `-1` means all `upper_val`
    /// - `lower_val` specifies the value to fill the lower triangular part with
    /// - `upper_val` specifies the value to fill the upper triangular part with
    pub fn tri(
        matrix_dim: usize,
        num_matrices: usize,
        diag: i32,
        lower_val: T,
        upper_val: T,
    ) -> Result<Tensor<T>> {
        // We make one matrix and then just clone it
        let data = (0i32..matrix_dim as i32)
            .flat_map(|i| {
                if (i + diag).is_negative() {
                    vec![upper_val; matrix_dim]
                } else {
                    std::iter::repeat_n(lower_val, (i + diag + 1) as usize)
                        .chain(std::iter::repeat(upper_val))
                        .take(matrix_dim)
                        .collect::<Vec<T>>()
                }
            })
            .cycle()
            .take(num_matrices * matrix_dim * matrix_dim)
            .collect::<Vec<T>>();

        Tensor::<T>::new(vec![num_matrices, matrix_dim, matrix_dim].into(), data)
    }

    pub fn min_value(&self) -> T {
        self.data.iter().fold(T::MAX, |min, x| min.cmp_min(x))
    }

    pub fn try_map<F: Fn(&T) -> anyhow::Result<T>>(&self, f: F) -> anyhow::Result<Self> {
        Ok(Self {
            data: self
                .data
                .iter()
                .map(f)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }

    // slice on the third dimension.
    // start inclusive, end exclusive
    pub fn slice_3d(&self, start: usize, end: usize) -> Result<Self> {
        ensure!(self.shape.len() == 3);
        ensure!(start < self.shape[0]);
        ensure!(end <= self.shape[0]);
        let blocks = self.shape[1] * self.shape[2];
        let sliced = self.data[blocks * start..blocks * end].to_vec();
        let shape: Shape = vec![end - start, self.shape[1], self.shape[2]].into();

        Ok(Self {
            data: sliced,
            shape: shape.clone(),
            unpadded_shape: shape,
        })
    }

    // slice the tensor on the second dimension
    // dim2_start inclusive
    // dim2_end exclusive
    // TODO: refactor to take generic shape dimensions where to slice ... or just use burn API tensor
    pub fn slice_2d(&self, dim2_start: usize, dim2_end: usize) -> Result<Self> {
        ensure!(self.shape.len() == 2);
        let range = dim2_start * self.shape[1]..dim2_end * self.shape[1];
        let data = self.data[range].to_vec();
        let new_shape: Shape = vec![dim2_end - dim2_start, self.shape[1]].into();
        Ok(Self {
            data,
            shape: new_shape.clone(),
            unpadded_shape: new_shape,
        })
    }

    #[cfg(test)]
    pub fn any(shape: Shape) -> impl proptest::prelude::Strategy<Value = Self> {
        use proptest::prelude::*;
        let size = shape.product();
        let data = proptest::collection::vec(T::any(), size);
        data.prop_map(move |data| Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape.clone(),
        })
    }

    pub fn random(shape: &Shape) -> Self {
        Self::random_seed(shape, Some(crate::seed_from_env_or_rng()))
    }

    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub fn random_seed(shape: &Shape, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(crate::seed_from_env_or_rng); // Use provided seed or default
        let mut rng = <crate::StdRng as ark_std::rand::SeedableRng>::seed_from_u64(seed);
        let size = shape.product();
        let data = (0..size).map(|_| T::random(&mut rng)).collect();

        Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape.clone(),
        }
    }
}

impl<T: TensorTypeParam> Tensor<T> {
    #[cfg(test)]
    pub fn into_wrapped(self) -> WrappedTensor<T> {
        WrappedTensor::try_from(&self).unwrap()
    }

    #[cfg(test)]
    pub fn as_wrapped(&self) -> WrappedTensor<T> {
        WrappedTensor::try_from(self).unwrap()
    }
}

impl<T: Clone + Default> Tensor<T> {
    pub fn pad_1d(mut self, new_len: usize) -> Result<Self> {
        ensure!(
            self.shape.len() == 1,
            "pad_1d only works for 1d tensors, e.g. vectors"
        );
        self.data.resize(new_len, Default::default());
        self.shape[0] = new_len;
        Ok(self)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Default + std::ops::Mul<Output = T> + std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    /// Parse the shape as N,C,H,W
    /// if the tensor is 3d, for example the input could be 3d if there is only one batch, then
    /// it returns as if N = 1.
    pub fn get4d(&self) -> (usize, usize, usize, usize) {
        let (n_size, offset) = if self.shape.len() == 3 {
            (1, 0)
        } else {
            (self.shape.first().cloned().unwrap_or(1), 1)
        };
        let c_size = self.shape.get(offset).cloned().unwrap_or(1);
        let h_size = self.shape.get(1 + offset).cloned().unwrap_or(1);
        let w_size = self.shape.get(2 + offset).cloned().unwrap_or(1);

        (n_size, c_size, h_size, w_size)
    }

    /// Retrieves an element using (N, C, H, W) indexing
    pub fn get_at_4d(&self, n: usize, c: usize, h: usize, w: usize) -> Result<T> {
        ensure!(self.shape.len() <= 4);

        let (n_size, c_size, h_size, w_size) = self.get4d();

        ensure!(n < n_size);
        let flat_index = n * (c_size * h_size * w_size) + c * (h_size * w_size) + h * w_size + w;
        Ok(self.data[flat_index])
    }

    // 0-based indexing for compatibility with other libraries
    // ex: accessors = [3,2,1] => will retrieve element at index 1 + 2 * shape[0] + 3 * shape[0] * shape[1]
    pub fn get(&self, accessors: Vec<usize>) -> Result<T> {
        let flat_index = self.get_idx(accessors)?;
        Ok(self.data[flat_index])
    }
}

impl<T> Tensor<T>
where
    T: Copy + Clone + Send + Sync,
    T: Copy + Default + std::ops::Mul<Output = T> + std::iter::Sum,
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    // Pads a matrix `M` to `M'` so that matrix-vector multiplication with a flattened FFT-padded convolution output `X'`
    /// matches the result of multiplying `M` with the original convolution output `X`.
    ///
    /// The real convolution output `X` has dimensions `(C, H, W)`. However, when using FFT-based convolution,
    /// the output `X'` is padded to dimensions `(C', H', W')`, where `C'`, `H'`, and `W'` are the next power of 2
    /// greater than or equal to `C`, `H`, and `W`, respectively.
    /// Given a matrix `M` designed to multiply with the flattened `X`, this function pads `M` into `M'` such that
    /// `M * X == M' * X'`, ensuring the result remains consistent despite the padding in `X'`.
    pub fn pad_matrix_to_ignore_garbage(
        &self,
        conv_shape_og: &[usize],
        conv_shape_pad: &[usize],
        mat_shp_pad: &Shape,
    ) -> Result<Self> {
        ensure!(
            conv_shape_og.len() == 3 && conv_shape_pad.len() == 3,
            "Expects conv2d shape output to be 3d: conv_shape_og: {:?}, conv_shape_pad: {:?}",
            conv_shape_og.len(),
            conv_shape_pad.len()
        );
        ensure!(
            mat_shp_pad.len() == 2 && self.shape.len() == 2,
            "Expects matrix to be 2d: mat_shp_pad: {:?}, self.shape: {:?}",
            mat_shp_pad.len(),
            self.shape.len()
        );
        let mat_shp_og = self.shape();

        let new_data: Vec<T> = (0..mat_shp_pad[0] * mat_shp_pad[1])
            .into_par_iter()
            .map(|new_loc| {
                // Decompose new_loc into (row, channel, h_in, w_in) for the padded output space
                let row = new_loc / mat_shp_pad[1];
                let channel =
                    (new_loc / (conv_shape_pad[1] * conv_shape_pad[2])) % conv_shape_pad[0];
                let h_in = (new_loc / conv_shape_pad[2]) % conv_shape_pad[1];
                let w_in = new_loc % conv_shape_pad[2];

                // Check if this position corresponds to an original data location
                if row < mat_shp_og[0]
                    && channel < conv_shape_og[0]
                    && h_in < conv_shape_og[1]
                    && w_in < conv_shape_og[2]
                {
                    let old_loc = channel * conv_shape_og[1] * conv_shape_og[2]
                        + h_in * conv_shape_og[2]
                        + w_in
                        + row * mat_shp_og[1];
                    self.data[old_loc]
                } else {
                    T::default() // Default value for non-mapped positions
                }
            })
            .collect();
        Tensor::new_with_unpadded_shape(
            mat_shp_pad.to_vec().into(),
            vec![self.unpadded_shape[0], mat_shp_pad[1]].into(),
            new_data,
        )
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shape = self.shape.clone();

        while shape.len() < 4 {
            shape.reverse();
            shape.push(1);
            shape.reverse();
        }

        if shape.len() == 4 {
            let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let channel_size = height * width;
            let batch_size = channels * channel_size;

            for b in 0..batches {
                writeln!(f, "Batch {b} [{channels} channels, {height}x{width}]:")?;
                for c in 0..channels {
                    writeln!(f, "  Channel {c}:")?;
                    let offset = b * batch_size + c * channel_size;
                    for i in 0..height {
                        let row_start = offset + i * width;
                        let row_data: Vec<String> = (0..width)
                            .map(|j| format!("{:>4.2}", self.data[row_start + j]))
                            .collect();
                        writeln!(f, "    {:>3}: [{}]", i, row_data.join(", "))?;
                    }
                }
            }
            write!(f, "Shape: {:?}", self.shape)
        } else {
            write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data) // Fallback
        }
    }
}

impl PartialEq for Tensor<Element> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl PartialEq for Tensor<f32> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl PartialEq for Tensor<GoldilocksExt2> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

pub struct TensorSlice<'a, T> {
    data: &'a [T],
    shape: Shape,
}

impl<'a, T> From<&'a Tensor<T>> for TensorSlice<'a, T> {
    fn from(value: &'a Tensor<T>) -> Self {
        Self {
            data: &value.data,
            shape: value.shape.clone(),
        }
    }
}

impl<'a, T> TensorSlice<'a, T> {
    pub(crate) fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    pub(crate) fn get_data(&self) -> &[T] {
        self.data
    }

    pub(crate) fn slice_over_first_dim(&self, dim2_start: usize, dim2_end: usize) -> Self {
        let range = dim2_start * self.shape[1]..dim2_end * self.shape[1];
        let data = &self.data[range];
        let mut new_shape = self.shape.clone();
        new_shape[0] = dim2_end - dim2_start;
        Self {
            data,
            shape: new_shape,
        }
    }
}

impl<'a, T: Clone> TensorSlice<'a, T> {
    pub(crate) fn to_tensor(&self) -> Result<Tensor<T>> {
        Tensor::new(self.shape.clone(), self.data.to_vec())
    }
}

impl<T: Default + Clone + Copy> Tensor<T> {
    /// Permute a tensor.
    ///
    /// The tensor's dimensions will be moved according to `order`. The `i`-th
    /// entry in the `order` vector specifies which dimension of the original
    /// tensor should become the `i`-th dimension of the output tensor.
    ///
    /// ```
    /// # use zkml::{Tensor, Shape};
    /// let tensor = Tensor::<i64>::random(&Shape::new(vec![2,3,5]));
    /// let permuted = tensor.permute3d(&[2,1,0]).unwrap();
    /// assert_eq!(tensor.dim(0), permuted.dim(2));
    /// assert_eq!(tensor.dim(1), permuted.dim(1));
    /// assert_eq!(tensor.dim(2), permuted.dim(0));
    /// ```
    pub fn permute3d(&self, order: &[usize]) -> Result<Self> {
        ensure!(
            self.rank() == 3,
            "Current tensor must be 3D. got {}",
            self.rank(),
        );
        ensure!(
            order.len() == 3,
            "New order must be 3D. got {}",
            order.len(),
        );
        let count = order.iter().filter(|x| **x < 3).sorted().dedup().count();
        ensure!(
            count == 3,
            "Order must have unique elements 0, 1, 2. got {order:?}",
        );

        // Special case, do nothing
        if order == [0, 1, 2] {
            return Ok(self.clone());
        }

        let new_shape = Shape::new(vec![
            self.dim(order[0]),
            self.dim(order[1]),
            self.dim(order[2]),
        ]);

        // reverse map from old position of a dimension to its new position
        let old_to_new = [
            order.iter().position(|v| *v == 0).unwrap(),
            order.iter().position(|v| *v == 1).unwrap(),
            order.iter().position(|v| *v == 2).unwrap(),
        ];
        let new_strides = new_shape.strides();
        let reverse_strides = [
            new_strides[old_to_new[0]],
            new_strides[old_to_new[1]],
            new_strides[old_to_new[2]],
        ];

        let (a, b, c) = (self.dim(0), self.dim(1), self.dim(2));
        let mut pos = 0;
        let mut data = vec![T::default(); self.shape.numel()];

        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    let new_loc =
                        i * reverse_strides[0] + j * reverse_strides[1] + k * reverse_strides[2];

                    data[new_loc] = self.data[pos];
                    pos += 1;
                }
            }
        }

        Ok(Self {
            data,
            shape: new_shape,
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }
}

impl<T: Copy> Tensor<T> {
    /// Instantiate a new tensor with `shape` initialised to `default`.
    pub fn initialised(shape: Shape, default: T) -> Self {
        Tensor {
            data: vec![default; shape.numel()],
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }
}

impl<T: Copy + Default> Tensor<T> {
    /// Copies the sub-slice `new_shape` from this tensor.
    ///
    /// Returns a new [Tensor] with shape `new_shape` initialised from `self`.
    pub fn reduce_to_shape(&self, new_shape: &Shape) -> Result<Self> {
        ensure!(
            self.rank() >= new_shape.rank(),
            "The target shape must be smaller than the current",
        );
        ensure!(
            self.shape()
                .iter()
                .rev()
                .zip(new_shape.iter().rev())
                .all(|(from, to)| from >= to),
            "The target dimensions can not be larger than the current",
        );
        // current position being copied
        let mut coord: Vec<_> = vec![0; new_shape.len()];
        // number of copies is equal to the number of elements in the target shape
        let mut copies = new_shape.numel();
        // auxiliary variable to convert from target to source coordinates
        let source_strides = self.shape.strides();
        let target_strides = new_shape.strides();

        let mut result = Tensor::initialised(new_shape.clone(), T::default());

        while copies > 0 {
            let source: usize = coord
                .iter()
                .zip(source_strides.iter())
                .map(|(pos, source)| pos * source)
                .sum();
            let dest: usize = coord
                .iter()
                .zip(target_strides.iter())
                .map(|(pos, target)| pos * target)
                .sum();

            result.data[dest] = self.data[source];
            copies -= 1;

            let mut rank = coord.len();
            while rank > 0 {
                rank -= 1;
                coord[rank] += 1;

                if coord[rank] == new_shape[rank] {
                    coord[rank] = 0;
                } else {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Pads the tensor to the next power-of-two.
    pub fn pad_next_power_of_two(&self) -> Self {
        let new_shape = self.shape().next_power_of_two();

        // Pre allocate the necessary capacity
        let mut new_data = Vec::with_capacity(new_shape.numel());
        new_data.extend(&self.data);

        // Create the new tensor and use the in-place implementation to change the shape
        let mut new_tensor = Tensor {
            data: new_data,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        };
        new_tensor.pad_to_shape(new_shape).expect(
            "padding within pad_to_next_power_of_two always \
            creates a new shape of correct rank; qed",
        );

        new_tensor
    }

    /// Pads the tensor to the next power-of-two using the specified value for padding.
    pub fn pad_next_power_of_two_with_value(&self, value: T) -> Self {
        let new_shape = self.shape().next_power_of_two();

        // Pre allocate the necessary capacity
        let mut new_data = Vec::with_capacity(new_shape.numel());
        new_data.extend(&self.data);

        // Create the new tensor and use the in-place implementation to change the shape
        let mut new_tensor = Tensor {
            data: new_data,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        };
        new_tensor.pad_to_shape_with_value(new_shape, value).expect(
            "padding within pad_to_next_power_of_two always \
            creates a new shape of correct rank; qed",
        );

        new_tensor
    }

    /// Changes the shape of the current [Tensor] to `target_shape`.
    ///
    /// This method will modify the current tensor in place, extending it
    /// to comply with the new shape.
    ///
    /// # Panics
    ///
    /// If the `target_shape` differs in rank or has a dimension smaller than
    /// the current tensor.
    pub fn pad_to_shape(&mut self, target_shape: Shape) -> Result<()> {
        self.pad_to_shape_with_value(target_shape, T::default())
    }

    /// Changes the shape of the current [Tensor] to `target_shape` by padding with the specified value.
    ///
    /// This method will modify the current tensor in place, extending it
    /// to comply with the new shape.
    ///
    /// # Panics
    ///
    /// If the `target_shape` differs in rank or has a dimension smaller than
    /// the current tensor.
    pub fn pad_to_shape_with_value(&mut self, target_shape: Shape, value: T) -> Result<()> {
        ensure!(
            target_shape.rank() == self.shape.rank(),
            "Target shape must have the same rank as the current tensor. current {:?} target {:?}",
            self.shape(),
            target_shape,
        );

        if self.shape == target_shape {
            return Ok(());
        }

        let distance = self
            .shape
            .iter()
            .zip(target_shape.iter())
            .map(|(original, new)| new.checked_sub(*original))
            .collect::<Option<Vec<usize>>>();

        let Some(distance) = distance else {
            bail!(
                "All dimensions of target shape must be greater-than-or-equal to the current tensor",
            );
        };

        debug_assert!(
            distance.iter().any(|v| *v != 0),
            "At least one dimensions must grow",
        );

        // At this point, the target_shape is known to be strictly bigger than the current shape.
        // And at least one of the dimensions has a non-zero distance

        // First expand the underlying storage vector to the new size
        self.data.resize(target_shape.product(), value);

        // Walks the shapes in reverse and count number of equal dimensions, if any.
        //
        // This determines the copy chunk size, when the dimensions don't
        // change no padding is necessary, so the data can be chunked together.
        let equal = distance.iter().rev().take_while(|v| **v == 0).count();
        let different = distance.len() - equal;
        debug_assert!(
            equal != target_shape.len(),
            "At least one dimensions must grown",
        );

        let strides = target_shape.strides();
        debug_assert_eq!(
            &strides[different..],
            &self.shape.strides()[different..],
            "The lower strides must be equal",
        );

        // Difference in size for a given dimension, i.e. how many empty spaces
        // are in between the dimensions after re-shaping.
        //
        // NOTE: this is non cumulative, each dimension padding is applied
        // separately.
        let mut padding = distance
            .iter()
            .zip(strides.iter())
            .map(|(distance, new)| distance * new)
            .take(different)
            .collect::<Vec<_>>();

        // the top-most dimension padding is implicit. This pop aligns the dimensions to remove off-by-one
        padding.remove(0);

        // Compute the chunk size for each copy
        let chunk = self
            .shape
            .iter()
            .rev()
            .take(equal + 1) // at least the last dimension can be copied in batches
            .product::<usize>();

        // The old data is being copied, so the number of copies is equal to the old shape
        let mut counters = self.shape.clone();
        debug_assert!(counters.len() > different - 1, "rank must not increase");
        counters.resize(different - 1, 0);

        // -1 because the very first entries are in the correct spot and don't need to be copied
        let mut copies = counters.iter().product::<usize>() - 1;

        let mut dest = target_shape.product();
        let mut end_pos = self.shape.product();

        // account for padding at the end
        dest -= distance
            .iter()
            .zip(strides)
            .map(|(dist, stride)| dist * stride)
            .sum::<usize>();

        // Copy original data into the new position, back to front to avoid overwrites
        while copies > 0 {
            copies -= 1;

            let start_pos = end_pos - chunk;
            dest -= chunk;

            self.data.copy_within(start_pos..end_pos, dest); // copy data to new position
            self.data[start_pos..min(end_pos, dest)].fill(value); // write the padding

            end_pos = start_pos;

            // Compute the new destination taking padding into consideration
            let mut dim = counters.len() - 1;
            loop {
                dest -= padding[dim];
                counters[dim] -= 1;

                if counters[dim] == 0 {
                    counters[dim] = self.shape[dim];
                    dim -= 1;
                } else {
                    break;
                }
            }
        }

        self.shape = target_shape;

        Ok(())
    }
}

impl<T: Default> Tensor<T> {
    fn get_idx(&self, accessors: Vec<usize>) -> Result<usize> {
        ensure!(self.shape.len() == accessors.len());
        let mut flat_index = *accessors.last().unwrap();
        let mut multiplier = *self.shape.last().unwrap();
        for (a, s) in accessors
            .iter()
            .rev()
            .skip(1)
            .zip(self.shape.iter().rev().skip(1))
        {
            ensure!(
                *a < *s,
                "Index out of bounds: {a} >= {s} - 0-based indexing forbids"
            );
            flat_index += *a * multiplier;
            multiplier *= *s;
        }
        Ok(flat_index)
    }
    pub fn map_data<O, F: Fn(&T) -> O>(&self, f: F) -> Tensor<O> {
        Tensor {
            data: self.data.iter().map(f).collect(),
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }

    pub fn insert_at_dim(mut self, dim: usize, index: usize, value: T) -> Result<Self> {
        if self.data.len() == index {
            self.data.push(value);
            *self.shape.get_mut(dim).unwrap() += 1;
        } else if self.data.len() > index {
            self.data[index] = value;
        } else {
            bail!(
                "Cannot insert at index {index} in tensor with data length {}",
                self.data.len()
            );
        }
        Ok(self)
    }

    /// The new shape of self will be [S1_1+S2_1,S1,...Sn]
    /// In other words, we only concatenate another vector if it's exactly size of the highest dimension
    pub fn concat_from_unpadded(
        &mut self,
        self_unpadded_first_dim: usize,
        other: Self,
        other_unpadded_first_dim: usize,
    ) -> anyhow::Result<()> {
        ensure!(
            self.shape().rank() == other.shape().rank(),
            "self and other shapes must have the same length"
        );
        ensure!(
            self.shape()
                .iter()
                .zip(other.shape().iter())
                .skip(1)
                .all(|(a, b)| a == b),
            "self and other shapes must have the same dimensions"
        );
        ensure!(
            other.shape().numel() == other.get_data().len(),
            "The other tensor data length is not equal to the other shape product"
        );

        // 0-based indexing
        let max_stride: usize = self.shape().iter().skip(1).product();
        let init_pos = self_unpadded_first_dim * max_stride;
        let mut pos = init_pos;
        let end_of_slice = self.get_data().len();
        // only take the non-padded part of the other tensor
        for new_v in other
            .data
            .into_iter()
            .take(other_unpadded_first_dim * max_stride)
        {
            if pos >= end_of_slice {
                self.data.push(new_v);
            } else {
                self.data[pos] = new_v;
            }
            pos += 1;
        }
        ensure!(
            pos.is_multiple_of(max_stride),
            "The part going beyond must be a multiple of the stride"
        );
        // how many times have we added a "big" chunk to the recipient, and therefore by how much to increase the new shape
        let new_dim = self_unpadded_first_dim + (pos - init_pos) / max_stride;
        // we only update the shape for the part that goes beyond the existing padding
        if new_dim > self.shape.dim(0) {
            self.shape.set_dim(0, new_dim);
            self.data.resize_with(self.shape.numel(), Default::default);
        }
        ensure!(
            self.get_data().len() == self.shape().product(),
            "The new data length {} is not equal to the new shape product {}",
            self.get_data().len(),
            self.shape().product()
        );
        Ok(())
    }
}

// taken from https://docs.pytorch.org/docs/stable/generated/torch.isclose.html
/// Determines whether two slices of `f32` values are element-wise close within
/// the specified absolute (`atol`) and relative (`rtol`) tolerances.
///
/// The condition checked is the same as PyTorch's `torch.isclose`:
/// `|a - b| <= atol + rtol * |b|` for every corresponding element.
///
/// # Examples
///
/// ```
/// use zkml::tensor::is_close_with_tolerance;
///
/// // For 10% relative tolerance (0.1 = 10%)
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.1, 2.2, 3.3]; // 10% difference
/// assert!(is_close_with_tolerance(&a, &b, 0.0, 0.1));
///
/// // For 1e-6 absolute tolerance
/// let c = [1.0, 2.0, 3.0];
/// let d = [1.000001, 2.000001, 3.000001]; // 1e-6 difference
/// assert!(is_close_with_tolerance(&c, &d, 1e-6, 0.0));
/// ```
pub fn is_close_with_tolerance(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(x, y)| {
        let diff = (*x - *y).abs();
        diff <= atol + rtol * y.abs()
    })
}

/// Backwards-compatible wrapper that uses the historical default tolerances
/// (`atol = 1e-8`, `rtol = 1e-5`).
pub fn is_close(a: &[f32], b: &[f32]) -> bool {
    is_close_with_tolerance(a, b, 1e-8_f32, 1e-5_f32)
}
/// Function used to get the broadcasted shape of two [Tensors][`Tensor`].
/// To be able to broadcast two [Tensors][`Tensor`] the shapes must be compatible in each dimension, this means either:
///     1) The two dimensions are equal
///     2) One of the dimensions is 1
/// If one [`Tensor`] has fewer dimensions than the other we can prepend its [`Shape`] with `1`s. For example if we have shapes
/// `[5, 7, 2]` and `[7, 1]` then let us write `[x, y, z]` for the currently unknown broadcasted shape, the broadcasting process works as follows:
///     1) Prepend `1` to the second [`Shape`] so that we have `[5, 7, 2]` and `[1, 7, 1]`
///     2) Compare the two shape arrays from back to front and apply our rules above:
///         i) `2` and `1` are not equal, but one of them is `1` so we set the final dim (`z`) in the broadcasted shape to `2` giving us `[x, y, 2]`
///         ii) Here both dims are `7` so because they are equal the broadcasted shape at `7` will also be `7` giving us `[x, 7, 2]`
///         iii) Finally we have `5` and `1`, since one of the dims is `1` we take the larger of the two giving `x = 5`
/// This gives us a final broadcasted shape of `[5, 7, 2]`.
///
/// As another example if we have a [`Tensor`] `A` with shape `[4, 1]` and values `[1, 2, 3, 4]` and we have another [`Tensor`] `B` with shape `[3]` and values `[10, 11, 12]`
/// then brodcasting them to the same shape results in:
/// ```ignore
///       A:  [1, 1, 1,       B:  [10, 11, 12
///            2, 2, 2,            10, 11, 12
///            3, 3, 3,            10, 11, 12
///            4, 4, 4]            10, 11, 12]
/// ```
pub fn get_broadcasted_shape(shape_a: &Shape, shape_b: &Shape) -> anyhow::Result<Shape> {
    // Compare the length of both inputs and match on the result
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();

    let compatibility = |a: usize, b: usize, index: usize| -> anyhow::Result<usize> {
        match (a, b) {
            // One of a or b is 1 so we return the value that is not 1
            (1, dim) | (dim, 1) => Ok(dim),
            // Both dims are the same so we return that value
            (dim_a, dim_b) if dim_a == dim_b => Ok(dim_a),
            // Any other case returns an error
            _ => Err(anyhow::anyhow!(
                "Cannot broadcast shapes as dimensions (a:{a}, b:{b}) were incompatible at index: {index}"
            )),
        }
    };

    match rank_a.cmp(&rank_b) {
        Ordering::Less => {
            // shape_a is shorter so we work out the difference and prepend with that many 1s
            let diff = rank_b - rank_a;

            let padded_shape_a = std::iter::repeat_n(1usize, diff)
                .chain(shape_a.iter().copied())
                .collect::<Vec<usize>>();

            // Now we iterate over both choosing the maximum each time
            shape_b
                .iter()
                .zip(padded_shape_a.iter())
                .enumerate()
                .map(|(index, (&b_dim, &a_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
        Ordering::Equal => {
            // The ranks are equal so we just iterate over both choosing the max each time
            shape_b
                .iter()
                .zip(shape_a.iter())
                .enumerate()
                .map(|(index, (&b_dim, &a_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
        Ordering::Greater => {
            // shape_a has larger rank so we prepend 1s to shape b
            let diff = rank_a - rank_b;

            let padded_shape_b = std::iter::repeat_n(1usize, diff)
                .chain(shape_b.iter().copied())
                .collect::<Vec<usize>>();
            // Now we iterate over both choosing the maximum each time
            shape_a
                .iter()
                .zip(padded_shape_b.iter())
                .enumerate()
                .map(|(index, (&a_dim, &b_dim))| compatibility(a_dim, b_dim, index))
                .collect::<Result<Vec<usize>, anyhow::Error>>()
                .map(Shape::from)
        }
    }
}

/// Checks `conv2d_tensor` and `fft_tensor` have the same result.
///
/// The contents of `conv2d_tensor` must come from a [Tensor::conv2d]
/// operation and have no padding. The `fff_tensor` must be the result of
/// [Tensor::fft_conv]. This utility will skip the garbage values of the fft.
///
/// expected is std conv2d (kw, nx-nw+1, nx-nw+1)
/// fft_tensor is results from fft conv (kw, nx, nx)
#[cfg(test)]
pub(crate) fn check_tensor_consistency(
    conv2d_tensor: &Tensor<Element>,
    fft_tensor: &Tensor<Element>,
) {
    assert_eq!(
        fft_tensor.shape().rank(),
        3,
        "FFT tensor should not have batching. shape {:?}",
        fft_tensor.shape(),
    );
    assert_eq!(
        conv2d_tensor.shape().rank(),
        3,
        "Tensor should not have batching. shape {:?}",
        conv2d_tensor.shape(),
    );
    assert_eq!(
        fft_tensor.shape()[1],
        fft_tensor.shape()[2],
        "FFT tensor should have same height and width. shape {:?}",
        fft_tensor.shape(),
    );
    assert!(
        fft_tensor.shape()[2].is_power_of_two(),
        "FFT tensor should have a power-of-two height and width. shape {:?}",
        fft_tensor.shape(),
    );

    let fft_strides = fft_tensor.shape().strides();
    let strides = conv2d_tensor.shape().strides();
    for channel in 0..conv2d_tensor.shape[0] {
        for height in 0..conv2d_tensor.shape[1] {
            for width in 0..conv2d_tensor.shape[2] {
                let expected_pos = channel * strides[0] + height * strides[1] + width * strides[2];
                let fft_pos =
                    channel * fft_strides[0] + height * fft_strides[1] + width * fft_strides[2];

                assert!(
                    conv2d_tensor.data[expected_pos] == fft_tensor.data[fft_pos],
                    "Error in tensor consistency. channel {channel} height {height} width {width} got {} expected {}",
                    fft_tensor.data[fft_pos],
                    conv2d_tensor.data[expected_pos],
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::Rng;
    use ff_ext::{FieldFrom, GoldilocksExt2};
    use ndarray::{Array, Ix2, Order};

    use crate::{
        rng_from_env_or_random,
        testing::{random_field_vector, random_vector},
        to_field,
    };

    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_bitreverse_permutation() {
        for logn in 1..10 {
            let n = 1 << logn;

            let mut expected: Vec<usize> = vec![0; n];
            for i in 1..n {
                expected[i] = expected[i >> 1] >> 1 | (i & 1) << (logn - 1);
            }

            for (n, i) in expected.iter().zip(bitreverse_permutation(n)) {
                assert_eq!(*n, i);
            }
        }
    }

    #[test]
    fn test_bitreverse() {
        for logn in 1..10 {
            let size = 1 << logn;
            let mut data = random_vector(size);

            let mut expected = data.clone();
            let bit_reverse = bitreverse_permutation(size).collect_vec();
            let permutation: Vec<(usize, usize)> = (0..size)
                .into_par_iter()
                .filter_map(|i| {
                    if bit_reverse[i] < i {
                        Some((i, bit_reverse[i]))
                    } else {
                        None
                    }
                })
                .collect();

            for (i, j) in permutation {
                expected.swap(i, j);
            }

            bitreverse(&mut data);
            assert_eq!(expected, data);
        }
    }

    #[test]
    fn test_tensor_basic_ops() {
        let tensor1 = Tensor::new(vec![2, 2].into(), vec![1, 2, 3, 4]).unwrap();
        let tensor2 = Tensor::new(vec![2, 2].into(), vec![5, 6, 7, 8]).unwrap();

        let result_add = tensor1.add(&tensor2);
        assert_eq!(
            result_add,
            Tensor::new(vec![2, 2].into(), vec![6, 8, 10, 12]).unwrap(),
            "Element-wise addition failed."
        );

        let result_sub = tensor2.sub(&tensor2);
        assert_eq!(
            result_sub,
            Tensor::zeros(vec![2, 2].into()),
            "Element-wise subtraction failed."
        );

        let result_mul = tensor1.mul(&tensor2);
        assert_eq!(
            result_mul,
            Tensor::new(vec![2, 2].into(), vec![5, 12, 21, 32]).unwrap(),
            "Element-wise multiplication failed."
        );

        let result_scalar = tensor1.scalar_mul(&2);
        assert_eq!(
            result_scalar,
            Tensor::new(vec![2, 2].into(), vec![2, 4, 6, 8]).unwrap(),
            "Element-wise scalar multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matvec() {
        let shape_m = vec![3, 3];
        let tensor_m =
            Tensor::new(shape_m.clone().into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let matrix = Array::from_vec(tensor_m.get_data().to_vec())
            .into_shape_with_order((shape_m, Order::RowMajor))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let tensor_v = Tensor::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        let vector = Array::from_vec(tensor_v.get_data().to_vec());

        let result = tensor_m.matvec(&tensor_v).unwrap();
        let expected_result = matrix.dot(&vector);

        assert_eq!(
            Array::from_vec(result.get_data().to_vec()),
            expected_result,
            "Matrix-vector multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matmul() {
        let shape_a = vec![4, 3];
        let tensor_a = Tensor::new(
            shape_a.clone().into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let matrix_a = Array::from_vec(tensor_a.get_data().to_vec())
            .into_shape_with_order((shape_a, Order::RowMajor))
            .unwrap();
        let shape_b = vec![3, 3];
        let tensor_b = Tensor::new(
            shape_b.clone().into(),
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
        .unwrap();
        let matrix_b = Array::from_vec(tensor_b.get_data().to_vec())
            .into_shape_with_order((shape_b, Order::RowMajor))
            .unwrap();

        let result = tensor_a.matmul(&tensor_b).unwrap();

        let expected_result = matrix_a
            .into_dimensionality::<Ix2>()
            .unwrap()
            .dot(&matrix_b.into_dimensionality::<Ix2>().unwrap());

        assert_eq!(
            Array::from_vec(result.get_data().to_vec())
                .into_shape_with_order((expected_result.shape(), Order::RowMajor))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            expected_result,
            "Matrix-matrix multiplication failed."
        );
    }

    #[test]
    fn test_tensor_transpose() {
        let matrix_a = Tensor::new(
            vec![3, 4].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let matrix_b = Tensor::new(
            vec![4, 3].into(),
            vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],
        )
        .unwrap();

        let result = matrix_a.transpose().unwrap();

        assert_eq!(result, matrix_b, "Matrix transpose failed.");
    }

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = Shape::new(vec![1, 1, 1, 1]);
        let tensor = Tensor::new(shape.clone(), vec![1]).unwrap();
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2]).unwrap();
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![4, 4]);
        let tensor = Tensor::<Element>::random(&shape.clone());
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![3, 3]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]).unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data(),
            [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![3, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2, 1, 2]).unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![4, 2]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data(),
            [1, 2, 1, 2, 1, 2, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![2, 3, 3]);
        let tensor = Tensor::new(
            shape.clone(),
            vec![
                1, 1, 1, 2, 2, 2, 3, 3, 3, 11, 11, 11, 12, 12, 12, 13, 13, 13,
            ],
        )
        .unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![2, 4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data(),
            [
                1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 0, 12, 12, 12, 0, 13,
                13, 13, 0, 0, 0, 0, 0,
            ],
            "Tensor padding to next power of two failed."
        );
    }

    impl Tensor<Element> {
        pub fn get_2d(&self, i: usize, j: usize) -> Element {
            assert!(self.shape.is_matrix());
            self.data[i * self.shape()[1] + j]
        }

        pub fn random_eval_point(&self) -> Vec<E> {
            let mut rng = rng_from_env_or_random();
            let r = rng.gen_range(0..self.nrows_2d().unwrap());
            let c = rng.gen_range(0..self.ncols_2d().unwrap());
            self.position_to_boolean_2d(r, c).unwrap()
        }
    }

    #[test]
    fn test_tensor_mle() {
        let mat = Tensor::random(&vec![3, 5].into());
        let shape = mat.shape();
        let mat = mat.pad_next_power_of_two();
        let mut mle = mat.to_2d_mle::<E>().unwrap();
        let mut rng = rng_from_env_or_random();
        let (chosen_row, chosen_col) = (rng.gen_range(0..shape[0]), rng.gen_range(0..shape[1]));
        let elem = mat.get_2d(chosen_row, chosen_col);
        let elem_field: E = elem.to_field();
        println!("(x,y) = ({chosen_row},{chosen_col}) ==> {elem:?}");
        let inputs = mat.position_to_boolean_2d(chosen_row, chosen_col).unwrap();
        let output = mle.evaluate(&inputs);
        assert_eq!(elem_field, output);

        // now try to address one at a time, and starting by the row, which is the opposite order
        // of the boolean variables expected by the MLE API, given it's expecting in LE format.
        let row_input = mat.row_to_boolean_2d(chosen_row).unwrap();
        mle.fix_high_variables_in_place(&row_input.collect_vec());
        let col_input = mat.col_to_boolean_2d(chosen_col).unwrap();
        let output = mle.evaluate(&col_input.collect_vec());
        assert_eq!(elem_field, output);
    }

    #[test]
    fn test_tensor_matvec_concatenate() {
        let matrix = Tensor::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let vector = Tensor::new(vec![3].into(), vec![10, 20, 30]).unwrap();

        let result = matrix.concat_matvec_col(&vector).unwrap();

        assert_eq!(
            result,
            Tensor::new(
                vec![3, 4].into(),
                vec![1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30]
            )
            .unwrap(),
            "Concatenate matrix vector as columns failed."
        );
    }

    type E = GoldilocksExt2;

    #[test]
    fn test_tensor_ext_ops() {
        let matrix_a_data = [1 as Element, 2, 3, 4, 5, 6, 7, 8, 9];
        let matrix_b_data = [10 as Element, 20, 30, 40, 50, 60, 70, 80, 90];
        let matrix_c_data = [300 as Element, 360, 420, 660, 810, 960, 1020, 1260, 1500];
        let vector_a_data = [10 as Element, 20, 30];
        let vector_b_data = [140 as Element, 320, 500];

        let matrix_a_data: Vec<E> = to_field(matrix_a_data);
        let matrix_b_data: Vec<E> = to_field(matrix_b_data);
        let matrix_c_data: Vec<E> = to_field(matrix_c_data);
        let vector_a_data: Vec<E> = to_field(vector_a_data);
        let vector_b_data: Vec<E> = to_field(vector_b_data);
        let matrix = Tensor::new(vec![3usize, 3].into(), matrix_a_data.clone()).unwrap();
        let vector = Tensor::new(vec![3usize].into(), vector_a_data).unwrap();
        let vector_expected = Tensor::new(vec![3usize].into(), vector_b_data).unwrap();

        let result = matrix.matvec(&vector).unwrap();

        assert_eq!(
            result, vector_expected,
            "Matrix-vector multiplication failed."
        );

        let matrix_a = Tensor::new(vec![3, 3].into(), matrix_a_data).unwrap();
        let matrix_b = Tensor::new(vec![3, 3].into(), matrix_b_data).unwrap();
        let matrix_c = Tensor::new(vec![3, 3].into(), matrix_c_data).unwrap();

        let result = matrix_a.matmul(&matrix_b).unwrap();

        assert_eq!(result, matrix_c, "Matrix-matrix multiplication failed.");
    }

    #[test]
    fn test_tensor_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 4].into(),
            vec![
                99, -35, 18, 104, -26, -48, -80, 106, 10, 8, 79, -7, -128, -45, 24, -91, -7, 88,
                -119, -37, -38, -113, -84, 86, 116, 72, -83, 100, 83, 81, 87, 58, -109, -13, -123,
                102,
            ],
        )
        .unwrap();
        let expected =
            Tensor::<Element>::new(vec![1, 3, 1, 2].into(), vec![99, 106, 88, 24, 116, 100])
                .unwrap();

        let result = input.maxpool2d(2, 2).unwrap();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
    }

    #[test]
    fn test_tensor_pad_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        )
        .unwrap();
        let expected = Tensor::<Element>::new(
            vec![1, 3, 2, 2].into(),
            vec![104, -1, 74, 3, 19, 70, 79, 21, 112, 65, 69, 105],
        )
        .unwrap();

        let padded_expected = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                104, 104, -1, -1, 104, 104, -1, -1, 74, 74, 3, 3, 74, 74, 3, 3, 19, 19, 70, 70, 19,
                19, 70, 70, 79, 79, 21, 21, 79, 79, 21, 21, 112, 112, 65, 65, 112, 112, 65, 65, 69,
                69, 105, 105, 69, 69, 105, 105,
            ],
        )
        .unwrap();

        let (result, padded_result) = input.padded_maxpool2d().unwrap();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
        assert_eq!(
            padded_result, padded_expected,
            "Padded Maxpool (Element) failed."
        );
    }

    #[test]
    fn test_pad_tensor_for_mle() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        )
        .unwrap();

        let padded = input.pad_next_power_of_two();

        padded
            .shape()
            .iter()
            .zip(input.shape().iter())
            .for_each(|(padded_dim, input_dim)| {
                assert_eq!(*padded_dim, input_dim.next_power_of_two())
            });

        let input_data = input.get_data();
        let padded_data = padded.get_data();
        for i in 0..1 {
            for j in 0..3 {
                for k in 0..4 {
                    for l in 0..4 {
                        let index = 3 * 4 * 4 * i + 4 * 4 * j + 4 * k + l;
                        assert_eq!(input_data[index], padded_data[index]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor_pad() {
        let shape_a = Shape::from_it([3, 1, 1]);
        let tensor_a = Tensor::<Element>::new(shape_a.clone(), vec![1; shape_a.product()]).unwrap();

        let shape_b = vec![4, 1, 1];
        let tensor_b = Tensor::<Element>::new(shape_b.clone().into(), vec![1, 1, 1, 0]).unwrap();

        let tensor_c = tensor_a.pad_next_power_of_two();
        assert_eq!(tensor_b, tensor_c);
    }

    #[test]
    fn test_tensor_pad_to_shape() {
        let shape = Shape::from_it([1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2]).unwrap();
        let target = Shape::from_it([3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([2, 1]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([1, 2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2, 2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2, 3, 4]).unwrap();
        let target = Shape::from_it([3, 3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0, 3, 4, 0, 0, 0, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 1]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 3]);
        let mut tensor =
            Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 3, 1]);
        let mut tensor =
            Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 2, 1, 3]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2]).unwrap();
        let target = Shape::from_it([2, 3, 5, 7]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                // x=0 y=0
                1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=1
                2, 2, 2, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=0
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=1
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);
    }

    #[test]
    fn test_tensor_conv2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
            ],
        )
        .unwrap();

        let weights = Tensor::<Element>::new(
            vec![2, 3, 2, 2].into(),
            vec![
                1, 0, -1, 2, 0, 1, -1, 1, 1, -1, 0, 2, -1, 1, 2, 0, 1, 0, 2, -1, 0, -1, 1, 1,
            ],
        )
        .unwrap();

        let bias = Tensor::<Element>::new(vec![2].into(), vec![3, -3]).unwrap();

        let expected = Tensor::<Element>::new(
            vec![1, 2, 2, 2].into(),
            vec![21, 22, 26, 27, 25, 25, 26, 26],
        )
        .unwrap();

        let result = input.conv2d(&weights, &bias, 1).unwrap();
        assert_eq!(result, expected, "Conv2D (Element) failed.");
    }

    #[test]
    fn test_tensor_minimal_conv2d() {
        // k_n,k_c,k_h,k_w
        let conv_shape = vec![2, 3, 3, 3].into();
        let conv = Tensor::<Element>::random(&conv_shape);
        // minimal input shape is 1,k_c,k_h,k_w
        let input_shape = vec![1, 3, 3, 3].into();
        let input = Tensor::<Element>::random(&input_shape);
        // minimal bias shape is k_n
        let bias = Tensor::<Element>::random(&vec![2].into());
        let output = input.conv2d(&conv, &bias, 1).unwrap();
        assert_eq!(*output.shape(), vec![1, 2, 1, 1].into());
    }

    #[test]
    fn test_tensor_pad_matrix_to_ignore_garbage() {
        let old_shape = Shape::new(vec![2usize, 3, 3]);
        let orows = 10usize;
        let ocols = old_shape.product();

        let new_shape = Shape::new(vec![3usize, 4, 4]);
        let nrows = 12usize;
        let ncols = new_shape.product();

        let og_t = Tensor::<Element>::random(&old_shape);
        let og_flat_t = og_t.to_flatten(); // This is equivalent to conv2d output (flattened)

        let mut pad_t = og_t.clone();
        pad_t.pad_to_shape(new_shape.clone()).unwrap();
        let pad_flat_t = pad_t.to_flatten();

        let og_mat = Tensor::random(&vec![orows, ocols].into()); // This is equivalent to the first dense matrix
        let og_result = og_mat.matvec(&og_flat_t).unwrap();

        let pad_mat = og_mat
            .pad_matrix_to_ignore_garbage(&old_shape, &new_shape, &vec![nrows, ncols].into())
            .unwrap();
        let pad_result = pad_mat.matvec(&pad_flat_t).unwrap();

        assert_eq!(
            og_result.get_data()[..orows],
            pad_result.get_data()[..orows],
            "Unable to get rid of garbage values from conv fft."
        );
    }

    #[test]
    fn test_tensor_slice_2d() {
        let tensor =
            Tensor::<Element>::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let sliced = tensor.slice_2d(0, 2).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3].into());
        assert_eq!(sliced.get_data(), vec![1, 2, 3, 4, 5, 6]);
        let sliced = tensor.slice_2d(2, 3).unwrap();
        assert_eq!(*sliced.shape(), vec![1, 3].into());
        assert_eq!(sliced.get_data(), vec![7, 8, 9]);
    }

    #[test]
    fn test_tensor_add_dim2() {
        let tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        let result = tensor.add_dim2(&vector);
        assert_eq!(*result.shape(), vec![2, 3].into());
        assert_eq!(result.get_data(), vec![11, 22, 33, 14, 25, 36]);
    }

    #[test]
    fn test_tensor_concat() {
        let mut tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        tensor.concat(vector).unwrap();

        assert_eq!(*tensor.shape(), vec![3, 3].into());
        assert_eq!(tensor.get_data(), vec![1, 2, 3, 4, 5, 6, 10, 20, 30]);

        let vector = Tensor::<Element>::new(vec![1, 3].into(), vec![66, 77, 88]).unwrap();
        tensor.concat(vector).unwrap();
        assert_eq!(*tensor.shape(), vec![4, 3].into());
        assert_eq!(
            tensor.get_data(),
            vec![1, 2, 3, 4, 5, 6, 10, 20, 30, 66, 77, 88]
        );
    }

    #[test]
    fn test_tensor_get() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();
        // 2 + 2 * 3 + 1 * 3 * 3 = 17
        assert_eq!(tensor.get(vec![1, 2, 2]).unwrap(), tensor.data[17]);
    }

    #[test]
    fn test_tensor_permute3d() {
        #[rustfmt::skip]
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,

                10, 11, 12,
                13, 14, 15,
                16, 17, 18,
            ],
        ).unwrap();

        let permuted = tensor.permute3d(&[1, 0, 2]).unwrap();
        assert_eq!(*permuted.shape(), vec![3, 2, 3].into());
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    let [new_i, new_j, new_k] = [j, i, k];
                    let expected = tensor.get(vec![i, j, k]).unwrap();
                    let given = permuted.get(vec![new_i, new_j, new_k]).unwrap();
                    assert_eq!(expected, given);
                }
            }
        }

        let tensor = Tensor::<Element>::random(&vec![18, 5, 27].into());
        let permuted = tensor.permute3d(&[1, 2, 0]).unwrap();
        assert_eq!(*permuted.shape(), Shape::new(vec![5, 27, 18]));
    }

    #[test]
    fn test_tensor_slice_3d() {
        let tensor = Tensor::<Element>::new(
            vec![3, 2, 2].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let sliced = tensor.slice_3d(1, 3).unwrap();
        assert_eq!(sliced.get_data(), vec![5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(*sliced.shape(), vec![2, 2, 2].into());
    }

    #[test]
    fn test_tensor_slices_last_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();

        let mut slices = tensor.slice_last_dim();

        // First slice
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        // Second slice
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        // Third slice
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        // Fourth slice
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        // Fifth slice
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        // Sixth slice
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        // No more slices
        assert_eq!(slices.next(), None);
    }

    #[test]
    fn test_tensor_slice_on_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();
        let (mut slices, shape) = tensor.slice_on_dim(1);
        assert_eq!(shape, Shape::new(vec![3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        assert_eq!(slices.next(), None);

        let (mut slices, shape) = tensor.slice_on_dim(0);
        assert_eq!(shape, Shape::new(vec![3, 3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            slices.next().unwrap(),
            &[10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        assert_eq!(slices.next(), None);

        let (slices, shape) = tensor.slice_on_dim(2);
        assert_eq!(shape, *tensor.shape());
        let data = slices.flatten().cloned().collect::<Vec<_>>();
        assert_eq!(
            data,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }

    #[test]
    fn test_tensor_argmax() {
        let tensor = Tensor::<Element>::new(vec![3].into(), vec![1, 2, 3]).unwrap();
        let argmax = tensor.argmax();
        assert_eq!(argmax, 2);
    }

    fn eval_lteq_poly(x_i: &[Element], y_i: &[Element]) -> Element {
        assert_eq!(x_i.len(), y_i.len());
        x_i.iter()
            .rev()
            .zip(y_i.iter().rev())
            .fold(Element::from(1), |acc, (x, y)| {
                acc * (1 - x - y + 2 * x * y) + (1 - x) * y
            })
    }

    fn eval_mle<F: ExtensionField + FieldFrom<u64>>(point: &[F]) -> F {
        let x_i = &point[..point.len() / 2];
        let y_i = &point[point.len() / 2..];
        x_i.iter().zip(y_i).fold(F::from_v(1), |acc, (&x, &y)| {
            acc * (F::from_v(1) - x - y + F::from_v(2) * x * y) + (F::from_v(1) - x) * y
        })
    }

    fn to_be_bits<const NUM_BITS: usize>(x: Element) -> [Element; NUM_BITS] {
        (0..NUM_BITS)
            .rev()
            .map(|i| {
                let mask = 1 << i;

                (x & Element::from(mask)) >> i
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[test]
    fn test_zeroifier_evaluation() {
        // create zeroifier matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let zeroifier_data = (0..num_columns * num_columns)
            .map(|i| {
                let r = i / num_columns;
                let c = i % num_columns;
                if r >= c {
                    Element::from(1)
                } else {
                    Element::from(0)
                }
            })
            .collect_vec();
        println!("Data: {zeroifier_data:?}");
        let zeroifier = Tensor::new(vec![num_columns, num_columns].into(), zeroifier_data).unwrap();
        assert_eq!(zeroifier.get_2d(0, 0), Element::from(1));
        assert_eq!(
            zeroifier.get_2d(num_columns - 1, num_columns - 1),
            Element::from(1)
        );
        assert_eq!(zeroifier.get_2d(0, 1), Element::from(0));
        assert_eq!(zeroifier.get_2d(1, 1), Element::from(1));
        assert_eq!(zeroifier.get_2d(1, 2), Element::from(0));

        let mle = zeroifier.to_2d_mle::<GoldilocksExt2>().unwrap();

        for i in 0..num_columns {
            for j in 0..num_columns {
                let x_i = to_be_bits::<NUM_BITS>(Element::from(i as u32));
                let y_i = to_be_bits::<NUM_BITS>(Element::from(j as u32));
                let cmp = eval_lteq_poly(&y_i, &x_i);
                assert_eq!(
                    zeroifier.get_2d(i, j),
                    cmp,
                    "Zeroifier evaluation failed for ({i}, {j})"
                );
                // build point for MLE: first column bits in little-endiian order, then rows bits in little-endian order
                let point = y_i
                    .into_iter()
                    .rev()
                    .chain(x_i.into_iter().rev())
                    .map(|bit| GoldilocksExt2::from_v(bit as u64))
                    .collect_vec();
                let eval = mle.evaluate(&point);
                assert_eq!(eval, GoldilocksExt2::from_v(cmp as u64));
                let quick_eval = eval_mle(&point);
                assert_eq!(eval, quick_eval);
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<GoldilocksExt2>(NUM_BITS * 2);
            assert_eq!(mle.evaluate(&point), eval_mle(&point),);
        }
    }

    #[test]
    fn test_tril() {
        // Test diag = 0
        let tensor = Tensor::<Element>::tril(4, 1, 0).unwrap();
        let real_value: Vec<Element> = vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(tensor.get_data(), real_value);
        // Test diag = 1
        let tensor = Tensor::<Element>::tril(4, 1, 1).unwrap();
        let real_value: Vec<Element> = vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(tensor.get_data(), real_value);
        // Test diag = -1
        let tensor = Tensor::<Element>::tril(4, 1, -1).unwrap();
        let real_value: Vec<Element> = vec![0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0];
        assert_eq!(tensor.get_data(), real_value);
    }

    #[test]
    fn test_concat_from() {
        let t1_shape: Shape = vec![5, 3, 3].into();
        let mut t1 = Tensor::<Element>::random(&t1_shape).pad_next_power_of_two();
        let t2_shape: Shape = vec![1, 3, 3].into();
        let t2 = Tensor::<Element>::random(&t2_shape).pad_next_power_of_two();
        t1.concat_from_unpadded(t1_shape.dim(0), t2, t2_shape.dim(0))
            .unwrap();
        // 8 since 5 padded next power of two is 8, and 5+1=6 so we don't go over the dimension
        let expected_shape = Shape::new(vec![8, 4, 4]);
        assert_eq!(*t1.shape(), expected_shape);
        // 6 + 3 = 9  so resulting
        let t3_shape: Shape = vec![3, 3, 3].into();
        let t3 = Tensor::<Element>::random(&t3_shape).pad_next_power_of_two();
        t1.concat_from_unpadded(6, t3, t3_shape.dim(0)).unwrap();
        let expected_shape = Shape::new(vec![9, 4, 4]);
        assert_eq!(*t1.shape(), expected_shape);
    }

    proptest! {
        #[test]
        fn test_pad_next_power_of_two_prop(x in 1usize..5, y in 2usize..=32, z in 2usize..=32) {
            pub fn pad_next_power_of_two(t: &Tensor<Element>) -> Tensor<Element> {
                let shape = t.shape();

                if shape.iter().all(|dim| dim.is_power_of_two()) {
                    return t.clone();
                }

                let padded_data = recursive_pad(t.get_data(), shape, Element::default());
                let padded_shape = shape.next_power_of_two();
                Tensor::new(padded_shape, padded_data).unwrap()
            }

            fn recursive_pad(data: &[Element], remaining_dims: &[usize], padding_value: Element) -> Vec<Element> {
                match remaining_dims.len() {
                    1 => data
                        .iter()
                        .cloned()
                        .chain(std::iter::repeat(padding_value))
                        .take(remaining_dims[0].next_power_of_two())
                        .collect::<Vec<_>>(),
                    _ => {
                        let chunk_size = remaining_dims[1..].iter().product::<usize>();
                        let mut unpadded_data = data
                            .par_chunks(chunk_size)
                            .map(|data_chunk| {
                                recursive_pad(data_chunk, &remaining_dims[1..], padding_value)
                            })
                            .collect::<Vec<Vec<_>>>();
                        let elem_size = unpadded_data[0].len();
                        unpadded_data.resize(
                            remaining_dims[0].next_power_of_two(),
                            vec![padding_value; elem_size],
                        );
                        unpadded_data.concat()
                    }
                }
            }

            let shape = Shape::new(vec![x, y, z]);
            let t = Tensor::<Element>::random(&shape);

            assert_eq!(t.pad_next_power_of_two(), pad_next_power_of_two(&t), "original: {t:?}");
        }

        #[test]
        fn proptest_tensor_permute3d(a in 2usize..=32, b in 2usize..=32, c in 2usize..=32) {
            fn permute3d<T: Default + Copy>(tensor: &Tensor<T>, order: &[usize]) -> Tensor<T> {
                let (a, b, c) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
                let new_a = tensor.shape[order[0]];
                let new_b = tensor.shape[order[1]];
                let new_c = tensor.shape[order[2]];

                let mut data = vec![T::default(); tensor.shape.numel()];
                for i in 0..a {
                    for j in 0..b {
                        for k in 0..c {
                            let old_loc = i * b * c + j * c + k;
                            let pos = [i, j, k];
                            let new_i = pos[order[0]];
                            let new_j = pos[order[1]];
                            let new_k = pos[order[2]];
                            let new_loc = new_i * new_b * new_c + new_j * new_c + new_k;
                            data[new_loc] = tensor.data[old_loc];
                        }
                    }
                }
                Tensor {
                    data,
                    shape: Shape::new(vec![new_a, new_b, new_c]),
                    unpadded_shape: Shape::new(vec![new_a, new_b, new_c]),
                }
            }

            let permutations = [
                [0, 1, 2],
                [1, 0, 2],
                [1, 2, 0],
                [0, 2, 1],
                [2, 0, 1],
                [2, 1, 0],
            ];

            let data = Tensor::<Element>::random(&Shape::new(vec![a, b, c]));
            for order in &permutations {
                let expected = permute3d(&data, order);
                let result = data.permute3d(order).unwrap();
                prop_assert_eq!(&expected, &result, "order {:?} original {:?}", order, data);
            }
        }

    }
}
