use crate::quantization::{self, MAX_FLOAT, MIN_FLOAT};
use anyhow::ensure;
use ark_std::rand::Rng;
use std::cmp::{Ordering, PartialEq};

use crate::Element;

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
    + std::fmt::Display
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

#[cfg(test)]
macro_rules! impl_number {
    ($field:ident) => {
        impl Number for $field {
            const MIN: $field = $field::ZERO;

            const MAX: $field = $field::ZERO;

            fn unit() -> Self {
                $field::ONE
            }

            fn zero() -> Self {
                $field::ZERO
            }

            fn random<R: Rng>(rng: &mut R) -> Self {
                $field::rand(rng)
            }

            fn absolute_value(&self) -> Self {
                *self
            }

            fn compare(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }

            fn is_negative(&self) -> bool {
                *self >= $field::from($field::MODULUS_MINUS_ONE_DIV_TWO)
            }

            fn to_f32(&self) -> anyhow::Result<f32> {
                unreachable!("Cannot convert field element to f32")
            }

            fn from_f32(_: f32) -> anyhow::Result<Self> {
                unreachable!("Cannot convert f32 to field element")
            }

            fn to_usize(&self) -> usize {
                let usize_max = $field::from(usize::MAX as u64);
                assert!(
                    *self <= usize_max,
                    "Canno convert to usize a field element bigger than `usize::MAX`"
                );
                let bigint = self.clone().into_bigint();
                let limb = bigint
                    .as_ref()
                    .iter()
                    .filter(|limb| **limb != 0)
                    .exactly_one()
                    .expect("More than 1 u64 limb found for field element");
                *limb as usize
            }

            fn from_usize(u: usize) -> Self {
                $field::from(u as u64)
            }

            #[cfg(test)]
            fn any() -> impl proptest::prelude::Strategy<Value = Self> {
                use proptest::prelude::Strategy;

                use crate::quantization::ToField;
                Element::any().prop_map(|el| el.to_field())
            }
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_ff::{AdditiveGroup, Field, PrimeField, UniformRand};
    use itertools::Itertools;

    type F = ark_bn254::Fr;
    #[cfg(test)]
    impl_number!(F);
}
