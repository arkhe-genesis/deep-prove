use burn::backend::NdArray;

use crate::Element;

use super::ZKMLBackend;

impl ZKMLBackend for NdArray<f32, Element> {}
