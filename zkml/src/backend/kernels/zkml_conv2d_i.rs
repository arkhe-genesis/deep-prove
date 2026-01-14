use cubecl::{prelude::*, std::FastDivmod};

/// Kernel to compute a 2d convolution.
///
/// This is a simplified convolution kernel, which does not support padding,
/// grouping, nor dilation. This kernel is invoked once per output element.
#[cube(launch)]
pub fn zkml_conv2d_i_kernel<I: Int>(
    input: &Tensor<I>,
    input_strides: Sequence<u32>,
    kernels: &Tensor<I>,
    kernels_strides: Sequence<u32>,
    bias: &Tensor<I>,
    output: &mut Tensor<I>,
    output_shape: Sequence<FastDivmod<u32>>,
    #[comptime] output_stride: u32,
) {
    // Handle extra kernels invocations due to cube alignment
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // decode the output position
    let (batch, rem) = output_shape.index(3_usize).div_mod(ABSOLUTE_POS as u32);
    let (o, rem) = output_shape.index(2_usize).div_mod(rem);
    let (oh, ow) = output_shape.index(1_usize).div_mod(rem);

    let (oh_stride, ow_stride) = (oh * output_stride, ow * output_stride);

    let mut sum = bias[o as usize];
    for channel in 0..kernels.shape(1) {
        for kh in 0..kernels.shape(2) {
            for kw in 0..kernels.shape(3) {
                let channel = channel as u32;
                let kh = kh as u32;
                let kw = kw as u32;

                let h = oh_stride + kh;
                let w = ow_stride + kw;

                // compute the input position
                let mut pos = batch * input_strides.index(3_usize);
                pos += channel * input_strides.index(2_usize);
                pos += h * input_strides.index(1_usize);
                pos += w;

                // compute the kernel position
                let mut kpos = o * kernels_strides.index(3_usize);
                kpos += channel * kernels_strides.index(2_usize);
                kpos += kh * kernels_strides.index(1_usize);
                kpos += kw;

                sum += input[pos as usize] * kernels[kpos as usize];
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}
