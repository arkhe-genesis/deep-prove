use cubecl::{prelude::*, std::FastDivmod};

/// Kernel to compute a 2d maxpool.
///
/// This is a simplified maxpool kernel, which does not support padding nor
/// dilation. This kernel is invoked once per output element.
#[cube(launch)]
pub fn zkml_max_pool2d_i_kernel<I: Int>(
    input: &Tensor<I>,
    input_strides: Sequence<u32>,
    output: &mut Tensor<I>,
    output_shape: Sequence<FastDivmod<u32>>,
    #[comptime] kernel_size: u32,
    #[comptime] output_stride: u32,
) {
    // Handle extra kernels invocations due to cube alignment
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // decode the output position
    let (batch, rem) = output_shape.index(3_usize).div_mod(ABSOLUTE_POS as u32);
    let (channel, rem) = output_shape.index(2_usize).div_mod(rem);
    let (oh, ow) = output_shape.index(1_usize).div_mod(rem);

    let (oh_stride, ow_stride) = (oh * output_stride, ow * output_stride);

    let mut max = I::min_value();
    #[unroll]
    for kw in 0..kernel_size {
        #[unroll]
        for kh in 0..kernel_size {
            let h = oh_stride + kh;
            let w = ow_stride + kw;

            // compute the input position
            let mut pos = batch * input_strides.index(3_usize);
            pos += channel * input_strides.index(2_usize);
            pos += h * input_strides.index(1_usize);
            pos += w;

            max = I::max(max, input[pos as usize]);
        }
    }

    output[ABSOLUTE_POS] = max;
}
