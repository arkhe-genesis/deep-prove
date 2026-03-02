//! Module containing code for quantising [`LayerNorm`] layers.

use super::*;

impl LayerNorm<f32> {
    /// Quantises the LayerNorm layer to use [`f16`](crate::number::f16) numbers.
    pub fn quantise(
        self,
        input_scaling_factor: ScalingFactor,
        normalisation_scaling_factor: ScalingFactor,
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<LayerNorm<Element>>> {
        // Now we construct the `model_scaling` from `self.gamma`
        let gamma_max = self.gamma.max_abs()?;
        let gamma_scaling = ScalingFactor::from_absolute_max(gamma_max, None);

        // Now work out the scaling factor for the beta tensor, which is used to quantize `self.beta`
        // This is calculated as `input_scaling * model_scaling * normalisation_scaling`
        let beta_scaling = gamma_scaling.scale() * normalisation_scaling_factor.scale();
        let pre_beta_bit_size =
            gamma_scaling.bit_size() + normalisation_scaling_factor.bit_size() + 1;

        let beta_domain_min: Element = -1 << (pre_beta_bit_size - 1);
        let beta_domain_max: Element = (1 << (pre_beta_bit_size - 1)) - 1;

        let beta_tensor = self.beta.wrapped_tensor()?.clone();
        let beta_tensor = Tensor::try_from(&beta_tensor)?;
        let beta_min = beta_tensor.min_value();
        let beta_max = beta_tensor.max();

        let beta_scaling_factor = ScalingFactor::from_parts(
            beta_max,
            beta_min,
            beta_scaling,
            (beta_domain_min, beta_domain_max),
        );

        let intermediate_bit_size = pre_beta_bit_size + 1;

        let quantised_gamma = self.gamma.quantize(&gamma_scaling);
        let quantised_beta = self.beta.quantize(&beta_scaling_factor);

        // Now we make the requant layer
        let multiplier = beta_scaling / output_scaling.scale();
        let requant = Requant::from_multiplier(multiplier, intermediate_bit_size, output_scaling);

        let quantised_layernorm = LayerNorm {
            gamma: quantised_gamma,
            beta: quantised_beta,
            eps: self.eps,
            normalisation_dim_size: self.normalisation_dim_size,
            mean_scaling_factor: Some(input_scaling_factor),
            normalisation_scaling_factor: Some(normalisation_scaling_factor),
            cache: Arc::new(Mutex::new(NormalisationCache::new())),
        };

        QuantizeOutput::new(quantised_layernorm, vec![output_scaling]).with_requant(requant)
    }
}
