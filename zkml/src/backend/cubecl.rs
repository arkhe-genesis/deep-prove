use std::{cmp::min, marker::PhantomData};

use anyhow::{Result, ensure};
use burn::{
    backend::{
        ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr},
        wgpu::{BoolElement, CubeBackend, FloatElement, IntElement},
    },
    tensor::{Shape as BShape, TensorMetadata, ops::IntTensor},
};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
use cubecl::{
    CubeCount, CubeDim,
    prelude::{ScalarArg, SequenceArg},
    std::FastDivmodArgs,
};

use crate::{Shape, backend::kernels};

use super::{Maxpool2dConfig, ZKMLBackend};

/// Returns a [CubeCount] that will perform at least `total` kernel invocations.
///
/// If `total` is a power-of-two this function will return an exact [CubeCount]
/// which calls the kernel the correct number of times, otherwise there will be
/// extra invocations.
fn fit_to_cube(total: u32, (max_x, max_y, max_z): (u32, u32, u32)) -> Result<CubeCount> {
    let maximum = max_x.saturating_mul(max_y.saturating_mul(max_z));
    ensure!(
        total <= maximum,
        "Request number of calls exceeds the maximum supported. requested {total} maximum {maximum}"
    );

    let mut x = total;
    let mut y = 1;
    let mut z = 1;

    fn bits(v: u32) -> u32 {
        u32::BITS - v.leading_zeros()
    }

    if x > max_x {
        // number of trailing zeros is equal to the exponent value for the factor 2.
        // these can be split across all dimensions with shifts without incurring
        // extra calls.
        let mut exp2 = x.trailing_zeros();
        let mut value = x >> exp2;

        // instead of factoring `value`, round the least significant bits, this
        // may incur into extra calls.
        let value_size = bits(value);
        let max_x_size = bits(max_x);
        let rounding_bits = value_size.saturating_sub(max_x_size);

        if rounding_bits > 0 {
            // perform an additional multiply-by-two when rounding is used.
            // this ensures the number of calls increases / rounds up.
            exp2 += rounding_bits + 1;
            value >>= rounding_bits;
        }

        if value > max_x {
            // value has a bit length that fits in max_x, however the value itself
            // is larger, divide an additional time by two.
            exp2 += 2;
            value >>= 1;
        } else {
            // value was eagerly divided by two, it may be smaller than max, adjust
            let value_num_bits = bits(value);
            let mut extra_bits = max_x_size.saturating_sub(value_num_bits);
            if (value << extra_bits) > max_x {
                extra_bits -= 1;
            }
            value <<= extra_bits;
            exp2 -= extra_bits;
        };

        let y_exp2 = min(max_y.ilog2(), exp2);
        exp2 -= y_exp2;
        x = value;
        y = 1 << y_exp2;
        z = 1 << exp2;

        // Rounding exceed the maximum number of calls, since the total was
        // checked to be within bounds at the start, use the maximum values
        // instead.
        if x * y * z > maximum {
            x = max_x;
            y = max_y;
            z = max_z;
        }
    }

    Ok(CubeCount::Static(x, y, z))
}

/// Computes the shape of the output tensor after performing the conv2d
fn conv2d_i_out_shape(
    input_dims: [usize; 4],
    kernels_dims: [usize; 4],
    config: &super::Conv2dConfig,
) -> Result<BShape> {
    // (N x C x H x W)
    let [batch_size, channels_in, height_in, width_in] = input_dims;
    // (M x C/group x kH x kW)
    let [feature_maps, channels_out, kernel_height, kernel_width] = kernels_dims;

    ensure!(
        channels_in == channels_out,
        "Grouping is currently not supported. channels in {channels_in} out {channels_out}",
    );

    // see [tensor::Tensor::conv2d] for details of the formula below
    let height_out = (height_in - kernel_height) / config.stride + 1;
    let width_out = (width_in - kernel_width) / config.stride + 1;
    Ok(BShape::new([
        batch_size,
        feature_maps,
        height_out,
        width_out,
    ]))
}

/// Computes the shape of the output tensor after performing the max_pool2d
fn max_pool2d_i_out_shape(input_dims: [usize; 4], config: &super::Maxpool2dConfig) -> BShape {
    // (N x C x H x W)
    let [batch_size, channels, height_in, width_in] = input_dims;

    // see [tensor::Tensor::maxpool2d] for details of the formula below
    let height_out = (height_in - config.kernel_size) / config.stride + 1;
    let width_out = (width_in - config.kernel_size) / config.stride + 1;
    BShape::new([batch_size, channels, height_out, width_out])
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> ZKMLBackend
    for CubeBackend<R, F, I, BT>
{
    fn zkml_conv2d_i(
        input: IntTensor<Self>,
        kernels: IntTensor<Self>,
        bias: IntTensor<Self>,
        config: super::Conv2dConfig,
    ) -> Result<IntTensor<Self>> {
        ensure!(input.shape.num_dims() == 4);
        ensure!(kernels.shape.num_dims() == 4);
        ensure!(bias.shape.num_dims() == 1);

        let shape_out = conv2d_i_out_shape(input.shape.dims(), kernels.shape.dims(), &config)?;

        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<I>());

        let output_strides = Shape::from(&shape_out).strides();
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
            I::dtype(),
        );

        // Because of the rounding done by div_ceil, it is possible for the kernel to be
        // called a few extra times. This is okay because the kernel handles out-of-bounds
        // calls.
        let cube_dim = CubeDim::default();
        let cube_count = fit_to_cube(
            (output.shape.num_elements() as u32).div_ceil(cube_dim.num_elems()),
            R::max_cube_count(),
        )?;

        let input_strides = Shape::from(&input.shape).strides();
        let input_strides = SequenceArg {
            values: vec![
                ScalarArg::new(input_strides[3] as u32),
                ScalarArg::new(input_strides[2] as u32),
                ScalarArg::new(input_strides[1] as u32),
                ScalarArg::new(input_strides[0] as u32),
            ],
        };
        let kernel_strides = Shape::from(&kernels.shape).strides();
        let kernel_strides = SequenceArg {
            values: vec![
                ScalarArg::new(kernel_strides[3] as u32),
                ScalarArg::new(kernel_strides[2] as u32),
                ScalarArg::new(kernel_strides[1] as u32),
                ScalarArg::new(kernel_strides[0] as u32),
            ],
        };
        let output_shape = SequenceArg {
            values: vec![
                FastDivmodArgs::new(&input.client, output_strides[3] as u32),
                FastDivmodArgs::new(&input.client, output_strides[2] as u32),
                FastDivmodArgs::new(&input.client, output_strides[1] as u32),
                FastDivmodArgs::new(&input.client, output_strides[0] as u32),
            ],
        };

        kernels::zkml_conv2d_i::zkml_conv2d_i_kernel::launch::<I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<I>(1),
            input_strides,
            kernels.as_tensor_arg::<I>(1),
            kernel_strides,
            bias.as_tensor_arg::<I>(1),
            output.as_tensor_arg::<I>(1),
            output_shape,
            config.stride as u32,
        );

        Ok(output)
    }

    fn zkml_max_pool2d_i(
        input: IntTensor<Self>,
        config: Maxpool2dConfig,
    ) -> Result<IntTensor<Self>> {
        ensure!(input.shape.num_dims() == 4);
        let shape_out = max_pool2d_i_out_shape(input.shape.dims(), &config);

        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<I>());

        let output_strides = Shape::from(&shape_out).strides();
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
            I::dtype(),
        );

        // Because of the rounding done by div_ceil, it is possible for the kernel to be
        // called a few extra times. This is okay because the kernel handles out-of-bounds
        // calls.
        let cube_dim = CubeDim::default();
        let cube_count = fit_to_cube(
            (output.shape.num_elements() as u32).div_ceil(cube_dim.num_elems()),
            R::max_cube_count(),
        )?;

        let input_strides = Shape::from(&input.shape).strides();
        let input_strides = SequenceArg {
            values: vec![
                ScalarArg::new(input_strides[3] as u32),
                ScalarArg::new(input_strides[2] as u32),
                ScalarArg::new(input_strides[1] as u32),
                ScalarArg::new(input_strides[0] as u32),
            ],
        };

        let output_shape = SequenceArg {
            values: vec![
                FastDivmodArgs::new(&input.client, output_strides[3] as u32),
                FastDivmodArgs::new(&input.client, output_strides[2] as u32),
                FastDivmodArgs::new(&input.client, output_strides[1] as u32),
                FastDivmodArgs::new(&input.client, output_strides[0] as u32),
            ],
        };

        kernels::zkml_max_pool2d_i::zkml_max_pool2d_i_kernel::launch::<I, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<I>(1),
            input_strides,
            output.as_tensor_arg::<I>(1),
            output_shape,
            config.kernel_size as u32,
            config.stride as u32,
        );

        Ok(output)
    }
}

impl<B: FusionBackend + ZKMLBackend> ZKMLBackend for Fusion<B> {
    fn zkml_conv2d_i(
        input: IntTensor<Self>,
        kernels: IntTensor<Self>,
        bias: IntTensor<Self>,
        config: super::Conv2dConfig,
    ) -> Result<IntTensor<Self>> {
        /// Metadata needed to run the operation once scheduled.
        #[derive(Debug)]
        struct Conv2dIIR {
            pub input: TensorIr,
            pub kernels: TensorIr,
            pub bias: TensorIr,
            pub config: super::Conv2dConfig,
            pub out: TensorIr,
        }

        /// Operation description that can be register.
        #[derive(Debug)]
        struct Conv2dIOps<B: FusionBackend> {
            description: Conv2dIIR,
            _phantom: PhantomData<B>,
        }

        impl<B: FusionBackend + ZKMLBackend> Operation<B::FusionRuntime> for Conv2dIOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B>(&self.description.input);
                let kernels = handles.get_int_tensor::<B>(&self.description.kernels);
                let bias = handles.get_int_tensor::<B>(&self.description.bias);
                // Forwards the operation to the implementation above
                let output = <B as ZKMLBackend>::zkml_conv2d_i(
                    input,
                    kernels,
                    bias,
                    self.description.config.clone(),
                )
                .expect("should be able to execute conv2diops; qed");
                handles.register_int_tensor::<B>(&self.description.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&input);
        streams.tensor(&kernels);
        streams.tensor(&bias);

        let shape = conv2d_i_out_shape(input.shape().dims(), kernels.shape().dims(), &config)?;
        let out = input
            .client
            .tensor_uninitialized(shape.dims.clone().into(), input.dtype());

        let description = Conv2dIIR {
            input: input.clone().into_ir(),
            kernels: kernels.clone().into_ir(),
            bias: bias.clone().into_ir(),
            config,
            out: out.to_ir_out(),
        };

        out.client.clone().register(
            streams,
            OperationIr::Custom(CustomOpIr {
                id: "conv2di".to_string(),
                inputs: vec![input.into_ir(), kernels.into_ir(), bias.into_ir()],
                outputs: vec![out.to_ir_out()],
            }),
            Conv2dIOps::<B> {
                description,
                _phantom: PhantomData,
            },
        );

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use cubecl::CubeCount;

    use super::fit_to_cube;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_fit_to_cube_pow2_max(total in 1u32..1024, max in 4u32..10) {
            let max_x = 1 << max;
            let max_y = 1 << max;
            let max_z = 1 << max;
            match fit_to_cube(total, (max_x, max_y, max_z)).unwrap() {
                CubeCount::Static(x, y, z) => {
                    prop_assert!(
                        total <= x * y * z,
                        "cube must perform at least {total} calls. x {x} y {y} z {z} total {} max_x {max_x} max_y {max_y} max_z {max_z}",
                        x * y * z,
                    );
                    prop_assert!(x <= max_x, "dimension value must be smaller than maximum. x {x} max_x {max_x}");
                    prop_assert!(y <= max_y, "dimension value must be smaller than maximum. y {y} max_x {max_y}");
                    prop_assert!(z <= max_z, "dimension value must be smaller than maximum. z {z} max_x {max_z}");
                },
                CubeCount::Dynamic(_) => unreachable!("fit_to_cube only returns static counts"),
            }
        }
    }
}
