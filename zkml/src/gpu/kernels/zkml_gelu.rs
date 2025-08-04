use cubecl::prelude::*;

#[cube(launch)]
pub fn zkml_gelu_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = gelu(input[ABSOLUTE_POS]);
    }
}

#[cube]
fn gelu<F: Float>(x: Line<F>) -> Line<F> {
    let c = comptime!((2.0f32 / std::f32::consts::PI).sqrt());

    let x_cubed = x * x * x;
    let inner_term = Line::new(F::new(c)) * (x + Line::new(F::new(0.044715)) * x_cubed);
    Line::new(F::new(0.5)) * x * (Line::new(F::new(1.0)) + Line::tanh(inner_term))
}
