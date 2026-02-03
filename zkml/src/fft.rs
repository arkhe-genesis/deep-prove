use anyhow::ensure;
use ff_ext::ExtensionField;
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_goldilocks::Goldilocks;
use rayon::{
    iter::{IntoParallelRefMutIterator, ParallelIterator},
    prelude::ParallelSliceMut,
};

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
fn bitreverse_permutation(length: usize) -> impl Iterator<Item = usize> {
    let shift = usize::BITS - length.ilog2();
    (0..length).map(move |i| i.reverse_bits() >> shift)
}

/// Applies a bitreverse order to the slice.
///
/// This can be used to start a FFT using decimation in time or to finalise a
/// FFT with decimation in frequency.
fn bitreverse<T>(d: &mut [T]) {
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
pub fn fft<E: ExtensionField + Send + Sync>(v: &mut Vec<E>, flag: bool) -> anyhow::Result<()> {
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

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{
        fft::{bitreverse, bitreverse_permutation},
        testing::random_vector,
    };

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
}
