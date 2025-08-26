use crate::Element;

#[cfg(all(feature = "cpu", not(feature = "gpu")))]
pub type Backend = burn::backend::NdArray<f32, Element>;

#[cfg(feature = "gpu")]
pub type Backend = burn::backend::Wgpu<f32, Element>;
