use burn::tensor::{ElementConversion, Tensor as BTensor, TensorPrimitive, ops::FloatTensor};

use crate::Element;

#[cfg(feature = "gpu")]
mod cubecl;

#[cfg(feature = "gpu")]
mod kernels;

#[cfg(feature = "cpu")]
mod ndarray;

#[cfg(all(feature = "cpu", not(feature = "gpu")))]
pub type Backend = burn::backend::NdArray<f32, Element>;

#[cfg(feature = "gpu")]
pub type Backend = burn::backend::Wgpu<f32, Element>;

pub(crate) trait ZKMLBackend: burn::tensor::backend::Backend {
    /// Custom GeLU implementation
    fn zkml_gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // compute: tensor * tensor * tensor
        let c0 = Self::IntElem::from_elem(3);
        let cubed = Self::float_powi_scalar(tensor.clone(), c0);

        // compute: sqrt(2 / PI) * (tensor + 0.044715 * cubed)
        let c1 = Self::FloatElem::from_elem(0.044715);
        let inner0 = Self::float_mul_scalar(cubed, c1);
        let inner1 = Self::float_add(tensor.clone(), inner0);
        let c2 = Self::FloatElem::from_elem((2.0_f32 / std::f32::consts::PI).sqrt());
        let inner2 = Self::float_mul_scalar(inner1, c2);

        // compute: 1.0 + tanh(inner2)
        let inner3 = Self::float_tanh(inner2);
        let one = Self::FloatElem::from_elem(1.0_f32);
        let inner4 = Self::float_add_scalar(inner3, one);

        // compute: 0.5 * tensor
        let half = Self::FloatElem::from_elem(0.5_f32);
        let inner5 = Self::float_mul_scalar(tensor, half);

        Self::float_mul(inner4, inner5)
    }
}

pub(crate) fn zkml_gelu<B: ZKMLBackend>(tensor: BTensor<B, 1>) -> BTensor<B, 1> {
    let output = B::zkml_gelu(tensor.into_primitive().tensor());
    BTensor::from_primitive(TensorPrimitive::Float(output))
}
