use burn::{
    backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement},
    tensor::ops::FloatTensor,
};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim};

use super::{ZKMLBackend, kernels};

/// Returns a [CubeCount] that will perform at least `total` kernel invocations.
///
/// If `total` is a power-of-two this function will return an exact [CubeCount]
/// which calls the kernel the correct number of times, otherwise there will be
/// extra invocations.
fn fit_to_cube(total: u32, (max_x, max_y, max_z): (u32, u32, u32)) -> CubeCount {
    let mut x = total;
    let mut y = 1;
    let mut z = 1;

    if x > max_x {
        // First try to evenly divide the work
        let div = std::cmp::min(max_y.trailing_zeros(), x.trailing_zeros());
        x >>= div;
        y <<= div;
        let div = std::cmp::min(max_z.trailing_zeros(), x.trailing_zeros());
        x >>= div;
        z <<= div;

        // Handle situations where work couldnt be split evenly
        let mut diff = x.saturating_sub(max_x);
        if diff > 0 {
            // NOTE: An alternative implementation would:
            //
            // - factor `x` into primes
            // - divide `x` by a prime and multiple either `y` or `z` by the same amount
            // - only round if both `y` and `z` cant be multiplied by the prime (because
            //   it would exceed their maximum)
            //
            // The above would produce the least amount of extra kernel
            // calls, but would increase the complexity and cost of
            // scheduling. For simplicity sake this solution skips the
            // factoring, and rounds the additional work to the next power
            // of two.
            diff = diff.next_power_of_two();

            // Increase the number of kernel calls so that x is even and the work can be divided
            x |= x & (diff - 1);
            x += diff;

            let div = std::cmp::min(
                max_x.trailing_zeros() - u32::trailing_zeros(y),
                x.trailing_zeros(),
            );
            x >>= div;
            y <<= div;
            let div = std::cmp::min(
                max_y.trailing_zeros() - u32::trailing_zeros(z),
                x.trailing_zeros(),
            );
            x >>= div;
            z <<= div;
        }
    }

    CubeCount::Static(x, y, z)
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> ZKMLBackend
    for CubeBackend<R, F, I, BT>
{
    fn zkml_gelu(data: FloatTensor<Self>) -> FloatTensor<Self> {
        let buffer = data
            .client
            .empty(data.shape.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            data.client.clone(),
            data.device.clone(),
            data.shape.clone(),
            buffer,
            F::dtype(),
        );

        let input_len =
            u32::try_from(data.shape.num_elements()).expect("Num of elements must fit in a u32");
        let elem = F::as_elem_native_unchecked();
        let line_size = R::line_size_elem(&elem)
            .filter(|line_size| input_len % u32::from(*line_size) == 0)
            .max()
            .unwrap_or(1);

        // Because of the rounding done by div_ceil, it is possible for the kernel to be
        // called a few extra times. This is okay because the kernel handles out-of-bounds
        // calls.
        let cube_dim = CubeDim::default();
        let elems_per_cube = cube_dim.num_elems() * u32::from(line_size);
        let cube_count = fit_to_cube(input_len.div_ceil(elems_per_cube), R::max_cube_count());

        kernels::zkml_gelu::zkml_gelu_kernel::launch::<F, R>(
            &data.client,
            cube_count,
            cube_dim,
            data.as_tensor_arg::<F>(line_size),
            output.as_tensor_arg::<F>(line_size),
        );

        output
    }
}
