//! Module that takes care of (re)quantizing
mod metadata;
mod strategy;
use derive_more::From;
use ff_ext::{ExtensionField, SmallField};
use serde::{Deserialize, Serialize};
use std::{env, sync::LazyLock};
use tracing::warn;

use crate::{
    Element,
    number::Number,
    tensor::{Tensor, TensorSlice, is_close},
    to_field,
};
pub use metadata::ModelMetadata;
pub use strategy::{
    AbsoluteMax, InferenceObserver, InferenceTracker, InferenceTrackingMode, ScalingStrategy,
    ScalingStrategyKind,
};

// Get BIT_LEN from environment variable or use default value
pub static BIT_LEN: LazyLock<usize> = LazyLock::new(|| {
    env::var("ZKML_BIT_LEN")
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(8) // Default value if env var is not set or invalid
});

/// symmetric quantization range
pub static MIN: LazyLock<Element> = LazyLock::new(|| -(1 << (*BIT_LEN - 1)));
pub static MAX: LazyLock<Element> = LazyLock::new(|| (1 << (*BIT_LEN - 1)) - 1);
pub static RANGE: LazyLock<Element> = LazyLock::new(|| *MAX - *MIN);
pub static ZERO: LazyLock<Element> = LazyLock::new(|| 0);
pub const MIN_FLOAT: f32 = -1.0;
pub const MAX_FLOAT: f32 = 1.0;
pub const QUANTIZATION_RANGE: std::ops::RangeInclusive<f32> = MIN_FLOAT..=MAX_FLOAT;

/// Symmetric quantization scaling
/// go from float [-a;a] to int [-2^BIT_LEN;2^BIT_LEN]
/// S = (a - (-a)) / (2^{BIT_LEN-1}- (-2^{BIT_LEN-1})) = 2a / 2^BIT_LEN
#[derive(Debug, Clone, From, Copy, Serialize, Deserialize)]
pub struct ScalingFactor {
    min: f32,
    max: f32,
    scale: f32,
    quantized_domain: (Element, Element),
}

impl ScalingFactor {
    pub fn from_absolute_max(abs_max: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        Self::from_span(-(abs_max.abs()), abs_max.abs(), quantized_domain)
    }
    pub fn from_tensor<T: MinMax>(
        t: &Tensor<T>,
        quantized_domain: Option<(Element, Element)>,
    ) -> Self {
        let max_abs = t
            .data()
            .iter()
            .fold(T::zero(), |a, b| a.cmp_max(b.absolute_value()));
        Self::from_absolute_max(max_abs.to_f32(), quantized_domain)
    }

    pub fn from_span(min: f32, max: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        let quantized_domain = quantized_domain.unwrap_or((*MIN, *MAX));
        let scale = (max - min) / (quantized_domain.1 - quantized_domain.0) as f32;
        Self {
            min,
            max,
            scale,
            quantized_domain,
        }
    }
    // Initialize a scaling factor in such a way that `self.scale()` is equal to the `scale` value
    // provided as input.
    pub(crate) fn from_scale(scale: f32, quantized_domain: Option<(Element, Element)>) -> Self {
        let (min_quantized, max_quantized) = quantized_domain.unwrap_or((*MIN, *MAX));
        let max = scale / 2.0 * (max_quantized - min_quantized) as f32;
        let min = -(max.abs());
        Self {
            max,
            min,
            scale,
            quantized_domain: (min_quantized, max_quantized),
        }
    }

    /// Create a [`ScalingFactor`] from its constituent parts. Useful for operations like Softmax where its
    /// important to preserve its structure as a probability distribution.
    pub(crate) fn from_parts(
        max: f32,
        min: f32,
        scale: f32,
        quantized_domain: (Element, Element),
    ) -> ScalingFactor {
        Self {
            max,
            min,
            scale,
            quantized_domain,
        }
    }

    pub fn min(&self) -> f32 {
        self.min
    }

    pub fn max(&self) -> f32 {
        self.max
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn domain(&self) -> (Element, Element) {
        self.quantized_domain
    }
    /// M = S1 * S2 / S3
    pub fn m(&self, s2: &Self, s3: &Self) -> f32 {
        self.scale() * s2.scale() / s3.scale()
    }

    /// Derives the right shift to apply to values to requantize them
    /// M = S1 * S2 / S3 = 2^-n * eps
    /// n is the number of bits to shift right
    pub fn shift(&self, s2: &Self, s3: &Self) -> usize {
        (-self.m(s2, s3).log2()).ceil() as usize
    }

    /// Take a floating point number and quantize it to an BIT_LEN-bit integer
    /// S = (a - (-a)) / (2^{BIT_LEN-1}- (-2^{BIT_LEN-1})) = 2a / 2^BIT_LEN
    pub fn quantize(&self, value: &f32) -> Element {
        let scaled = (*value / self.scale()).round_ties_even() as Element;
        if scaled < self.quantized_domain.0 || scaled > self.quantized_domain.1 {
            warn!(
                "Quantized value {} from {} is out of range [{}, {}]",
                scaled, value, self.quantized_domain.0, self.quantized_domain.1
            );
        }
        scaled.clamp(self.quantized_domain.0, self.quantized_domain.1)
    }

    pub fn dequantize(&self, value: &Element) -> f32 {
        *value as f32 * self.scale()
    }
}

impl Default for ScalingFactor {
    fn default() -> Self {
        let default_scale = 2.0f32 / (*MAX - *MIN) as f32;
        Self {
            min: -1.0,
            max: 1.0,
            scale: default_scale,
            quantized_domain: (*MIN, *MAX),
        }
    }
}

// s = m *  2^-shift, it returns the shift and the multiplier
pub fn split_scale_into_multiplier(s: f32) -> (i32, f32) {
    let log_s = s.log2();
    let (shift, m) = ((-log_s.trunc()), 2.0f32.powf(log_s.fract()));

    assert!(
        is_close(&[m * (2f32.powf(-shift))], &[s]),
        "m * 2^shift != s -> m: {}, s: {}, shift: {}, m * 2^shift: {}",
        m,
        s,
        shift,
        m * 2f32.powf(-shift)
    );
    (shift as i32, m)
}

/// Returns the scaling factors for the main tensor and for the bias tensor. These are the "model" scaling factors, or
/// S2 in the formula S1 * S2 / S3.
pub fn model_scaling_factor_from_tensor_and_bias(
    input: &ScalingFactor,
    main: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
) -> (ScalingFactor, ScalingFactor) {
    let max_weight = main.max_abs();
    let max_value = if let Some(bias) = bias {
        let max_bias = bias.max_abs();
        max_weight.max(max_bias)
    } else {
        max_weight
    };
    let main_sf = ScalingFactor::from_absolute_max(max_value, None);
    let bias_sf = bias_scaling_matmul(input, &main_sf);
    (main_sf, bias_sf)
}

pub fn bias_scaling_matmul(input: &ScalingFactor, model: &ScalingFactor) -> ScalingFactor {
    let min_quantized = -(1 << (2 * (*BIT_LEN) - 1)) + 1;
    let max_quantized = (1 << (2 * (*BIT_LEN) - 1)) - 1;
    ScalingFactor::from_scale(
        input.scale() * model.scale(),
        Some((min_quantized, max_quantized)),
    )
}

/// Trait to convert an element to its extension field.
///
/// The [From] and [Into] traits can not be used because of the orphan rules, it
/// is not allowed to implement a foreign trait for a foreign type.
///
/// The trait is generic over `F` because there are more than one target field,
/// e.g. baby bear and goldilocks.
pub trait ToField<F> {
    fn to_field(&self) -> F;
}

impl<F: ExtensionField> ToField<F> for Element {
    fn to_field(&self) -> F {
        if self.is_negative() {
            // Doing wrapped arithmetic : p-128 ... p-1 means negative number
            F::from_canonical_u64(<F::BaseField as SmallField>::MODULUS_U64 - self.unsigned_abs())
        } else {
            // for positive and zero, it's just the number
            F::from_canonical_u64(*self as u64)
        }
    }
}

/// Trait to convert a field element to its base.
///
/// The [From] and [Into] traits can not be used because of the orphan rules, it
/// is not allowed to implement a foreign trait for a foreign type.
pub(crate) trait ToElement {
    fn to_element(&self) -> Element;
}

impl<F: ExtensionField> ToElement for F {
    fn to_element(&self) -> Element {
        let e = self.to_canonical_u64_vec()[0];
        let modulus_half = <F::BaseField as SmallField>::MODULUS_U64 >> 1;
        // That means he's a positive number
        if *self == F::ZERO {
            0
        // we dont assume any bounds on the field elements, requant might happen at a later stage
        // so we assume the worst case
        } else if e <= modulus_half {
            e as Element
        } else {
            // That means he's a negative number - so take the diff with the modulus and recenter around 0
            let diff = <F::BaseField as SmallField>::MODULUS_U64 - e;
            -(diff as Element)
        }
    }
}

impl<F: ExtensionField, T> ToField<Vec<F>> for [T]
where
    T: ToField<F>,
{
    fn to_field(&self) -> Vec<F> {
        to_field::<T, F, _>(self)
    }
}

impl<F: ExtensionField, T> ToField<Tensor<F>> for Tensor<T>
where
    T: ToField<F>,
{
    fn to_field(&self) -> Tensor<F> {
        let shape = self.shape();
        let unpadded_shape = self.unpadded_shape();
        let data = to_field::<T, F, _>(self.data());
        Tensor::new_with_unpadded_shape(shape.clone(), unpadded_shape.clone(), data)
            .expect("The data length hasn't changed so this should never fail")
    }
}

impl<'a, F: ExtensionField, T> ToField<Tensor<F>> for TensorSlice<'a, T>
where
    T: ToField<F>,
{
    fn to_field(&self) -> Tensor<F> {
        Tensor::new(self.get_shape(), to_field::<T, F, _>(self.get_data())).expect(
            "a tensor can always be created from previously\
            successfully created shape and data; qed",
        )
    }
}

pub trait Quantize {
    type Output;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output;
}

impl Quantize for Element {
    type Output = Element;

    fn quantize(&self, _scaling: &ScalingFactor) -> Self::Output {
        *self
    }
}

impl Quantize for f32 {
    type Output = Element;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        scaling.quantize(self)
    }
}

impl<T> Quantize for [T]
where
    T: Quantize,
{
    type Output = Vec<<T as Quantize>::Output>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        self.iter().map(|e| T::quantize(e, scaling)).collect()
    }
}

impl<T> Quantize for Vec<T>
where
    T: Quantize,
{
    type Output = Vec<<T as Quantize>::Output>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        self.iter().map(|e| T::quantize(e, scaling)).collect()
    }
}

pub trait Dequantize {
    type Output;

    fn dequantize(&self, scaling: &ScalingFactor) -> Self::Output;
}

impl Dequantize for Element {
    type Output = f32;

    fn dequantize(&self, scaling: &ScalingFactor) -> Self::Output {
        scaling.dequantize(self)
    }
}

impl<T> Dequantize for [T]
where
    T: Dequantize,
{
    type Output = Vec<<T as Dequantize>::Output>;

    fn dequantize(&self, scaling: &ScalingFactor) -> Self::Output {
        self.iter().map(|el| el.dequantize(scaling)).collect()
    }
}

pub fn max_range_from_weight<T: Number>(weight: &T, min_input: &T, max_input: &T) -> (T, T) {
    let min = if weight.is_negative() {
        *weight * *max_input
    } else {
        *weight * *min_input
    };
    let max = if weight.is_negative() {
        *weight * *min_input
    } else {
        *weight * *max_input
    };
    (min, max)
}

pub trait MinMax {
    fn zero() -> Self;
    fn absolute_value(&self) -> Self;
    fn cmp_max(&self, other: Self) -> Self;
    fn to_f32(&self) -> f32;
}

impl MinMax for f32 {
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn zero() -> Self {
        0.0
    }
    fn cmp_max(&self, other: Self) -> Self {
        self.max(other)
    }
    fn to_f32(&self) -> f32 {
        *self
    }
}

impl MinMax for Element {
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn cmp_max(&self, other: Self) -> Self {
        std::cmp::max(*self, other)
    }
    fn zero() -> Self {
        0
    }
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

#[cfg(test)]
mod test {
    use crate::quantization::{ToElement, ToField, split_scale_into_multiplier};

    use crate::Element;

    use super::{MAX, MIN};
    type F = ff_ext::GoldilocksExt2;

    #[test]
    fn test_wrapped_arithmetic() {
        #[derive(Clone, Debug)]
        struct TestCase {
            a: Element,
            b: Element,
            res: Element,
        }

        let cases = [
            TestCase {
                a: -53,
                b: 10,
                res: -53 * 10,
            },
            TestCase {
                a: -45,
                b: -56,
                res: 45 * 56,
            },
        ];
        for (i, case) in cases.iter().enumerate() {
            // cast them to handle overflow
            let ap: F = case.a.to_field();
            let bp: F = case.b.to_field();
            let res = ap * bp;
            let expected = case.res.to_field();
            assert_eq!(res, expected, "test case {i}: {case:?}");
        }
    }

    #[test]
    fn test_element_field_roundtrip() {
        // Also test a few specific values explicitly
        let test_values = [*MIN, -100, -50, -1, 0, 1, 50, 100, *MAX];
        for &val in &test_values {
            let field_val: F = val.to_field();
            let roundtrip = field_val.to_element();

            assert_eq!(
                val, roundtrip,
                "Element {val} did not roundtrip correctly (got {roundtrip})"
            );
        }
    }

    #[test]
    fn test_split_scale_into_multiplier() {
        for s in [0.125, 0.075] {
            let (shift, m) = split_scale_into_multiplier(s);

            assert!((m * 2f32.powf(-shift as f32) - s).abs() <= f32::EPSILON);
        }
    }
}
