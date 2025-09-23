use crate::quantization::{self, MAX_FLOAT, MIN_FLOAT};
use anyhow::ensure;
use ark_std::rand::Rng;
use ceno_p3::field::FieldAlgebra;
use ff_ext::GoldilocksExt2;
use std::cmp::{Ordering, PartialEq};

use crate::{Element, quantization::Fieldizer};

pub trait Number:
    Copy
    + PartialEq
    + Clone
    + Send
    + Sync
    + Default
    + std::iter::Sum
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Mul<Output = Self>
    + std::fmt::Debug
{
    const MIN: Self;
    const MAX: Self;
    fn unit() -> Self;
    fn zero() -> Self;
    fn random<R: Rng>(rng: &mut R) -> Self;
    /// reason abs is necessary is because f32 doesn't implement Ord trait, so to have uniform code for f32 and Element,
    /// we implement abs here.
    fn absolute_value(&self) -> Self;
    fn cmp_max(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *self,
            Ordering::Equal => *self,
            Ordering::Less => *other,
        }
    }
    fn cmp_min(&self, other: &Self) -> Self {
        match self.compare(other) {
            Ordering::Greater => *other,
            Ordering::Equal => *self,
            Ordering::Less => *self,
        }
    }
    fn compare(&self, other: &Self) -> Ordering;
    fn is_negative(&self) -> bool;
    fn to_f32(&self) -> anyhow::Result<f32>;
    fn from_f32(f: f32) -> anyhow::Result<Self>;
    fn to_usize(&self) -> usize;
    fn from_usize(u: usize) -> Self;

    #[cfg(test)]
    fn any() -> impl proptest::prelude::Strategy<Value = Self>;
}

impl Number for Element {
    const MIN: Element = Element::MIN;
    const MAX: Element = Element::MAX;
    fn unit() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(*quantization::MIN..=*quantization::MAX)
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
    fn is_negative(&self) -> bool {
        *self < 0
    }
    fn to_f32(&self) -> anyhow::Result<f32> {
        ensure!(
            *self >= f32::MIN.ceil() as Element,
            "Element {self} is smaller than the minimum integer representable by f32"
        );
        ensure!(
            *self <= f32::MAX.floor() as Element,
            "Element {self} is bigger than the maximum integer representable by f32"
        );
        Ok(*self as f32)
    }
    fn from_f32(f: f32) -> anyhow::Result<Self> {
        Ok(f as Element)
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
    fn from_usize(u: usize) -> Self {
        u as Element
    }

    #[cfg(test)]
    fn any() -> impl proptest::prelude::Strategy<Value = Self> {
        *quantization::MIN..=*quantization::MAX
    }
}

impl Number for f32 {
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    fn unit() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(MIN_FLOAT..=MAX_FLOAT)
    }
    fn absolute_value(&self) -> Self {
        self.abs()
    }
    fn compare(&self, other: &Self) -> Ordering {
        if self < other {
            Ordering::Less
        } else if self == other {
            Ordering::Equal
        } else {
            Ordering::Greater
        }
    }

    fn is_negative(&self) -> bool {
        *self < 0.0
    }
    fn to_f32(&self) -> anyhow::Result<f32> {
        Ok(*self)
    }
    fn from_f32(f: f32) -> anyhow::Result<Self> {
        Ok(f)
    }
    fn to_usize(&self) -> usize {
        *self as usize
    }
    fn from_usize(u: usize) -> Self {
        u as f32
    }

    #[cfg(test)]
    fn any() -> impl proptest::prelude::Strategy<Value = Self> {
        MIN_FLOAT..=MAX_FLOAT
    }
}

impl Number for GoldilocksExt2 {
    const MIN: GoldilocksExt2 = GoldilocksExt2::ZERO;
    const MAX: GoldilocksExt2 = GoldilocksExt2::ZERO;
    fn unit() -> Self {
        GoldilocksExt2::ONE
    }
    fn zero() -> Self {
        GoldilocksExt2::ZERO
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        Element::random(rng).to_field()
    }
    fn absolute_value(&self) -> Self {
        *self
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }

    fn is_negative(&self) -> bool {
        panic!("GoldilocksExt2: is_negative is meaningless");
    }

    fn to_f32(&self) -> anyhow::Result<f32> {
        unreachable!("Called to_f32 for Goldilocks")
    }
    fn from_f32(_: f32) -> anyhow::Result<Self> {
        unreachable!("Called from_f32 for Goldilocks")
    }
    fn to_usize(&self) -> usize {
        unreachable!("Called to_usize for Goldilocks")
    }
    fn from_usize(_: usize) -> Self {
        unreachable!("Called from_usize for Goldilocks")
    }

    #[cfg(test)]
    fn any() -> impl proptest::prelude::Strategy<Value = Self> {
        use proptest::prelude::Strategy;
        Element::any().prop_map(|el| el.to_field())
    }
}
