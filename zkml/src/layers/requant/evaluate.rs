//! Code for evaluating the Requant layer.

use super::*;

impl Requant {
    pub(crate) fn evaluate_internal(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element>> {
        let outputs = inputs
            .iter()
            .map(|wrapped_input| {
                self.activation_lookup_data
                    .evaluate((*wrapped_input).clone())
            })
            .collect::<Result<Vec<WrappedTensor<Element>>>>()?;
        Ok(LayerOut::from_vec(outputs))
    }
}
