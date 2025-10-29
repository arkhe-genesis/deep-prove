use crate::{
    Number,
    graph::NodeId,
    layers::{
        Layer,
        activation::{Activation, GeGlu},
        matrix_mul::MatMul,
    },
};
use anyhow::ensure;

use crate::{
    model::Model,
    parser::{
        json,
        llm::{LLMConfig, config::LLMStructure},
    },
    tensor::KeyedTensor,
};

#[derive(Debug, Clone)]
pub struct FeedForward<N: Number> {
    pub gate: Option<MatMul<N>>, // used only for a Gated Linear Unit (GLU)
    pub up: KeyedTensor<N>,
    pub up_bias: Option<KeyedTensor<N>>,
    pub down: KeyedTensor<N>,
    pub down_bias: Option<KeyedTensor<N>>,
}

impl FeedForward<f32> {
    pub fn write_to_model(
        self,
        _config: &LLMStructure,
        model: &mut Model<f32>,
        input_node_id: NodeId,
    ) -> anyhow::Result<NodeId> {
        let up = MatMul::new_constant(self.up, self.up_bias)?;

        // let down = MatMul::new_constant(self.down, self.down_bias);
        let down = MatMul::new_constant(self.down, self.down_bias)?;
        let up_node_id = model.add_consecutive_layer(Layer::MatMul(up), Some(input_node_id))?;
        let activation_node_id = if let Some(gate) = self.gate {
            // in this case, the input is processed though another linear layer (i.e., gate_linear),
            // which is then processed by the activation function and combined with the output of `up` linear
            // component. Combining the output of activation function with `up` is already done inside the
            // activation layer being instantiated
            let gate_node_id =
                model.add_consecutive_layer(Layer::MatMul(gate), Some(input_node_id))?;
            let geglu: GeGlu<f32> = Activation::new_geglu();
            let geglu_id = model.graph.add_inner(geglu.into())?;
            // build input wires for GeGlu
            model.add_edge(gate_node_id, geglu_id, (0, GeGlu::<f32>::GELU_INPUT_INDEX))?;
            model.add_edge(up_node_id, geglu_id, (0, GeGlu::<f32>::LINEAR_INPUT_INDEX))?;
            geglu_id
        } else {
            // if there is no `self.gate`, then we just feed output of `up` linear component to the activation layer
            model.add_consecutive_layer(
                Layer::Activation(Activation::new_gelu()),
                Some(up_node_id),
            )?
        };
        let last_node_id =
            model.add_consecutive_layer(Layer::MatMul(down), Some(activation_node_id))?;
        Ok(last_node_id)
    }

    /// only supports gpt2 for now
    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let up = l.get_tensor("ffn_up.weight")?;
        let up_bias = l.get_tensor("ffn_up.bias")?;
        let down = l.get_tensor("ffn_down.weight")?;
        let down_bias = l.get_tensor("ffn_down.bias")?;
        ensure!(
            up.shape()[0] == c.hidden_size,
            "up have shape {:?} but in features should be equal to hidden_size: {}",
            up.shape(),
            c.hidden_size
        );
        ensure!(
            down.shape()[1] == c.embedding_size,
            "down have shape {:?} but out features should be equal to embedding_size: {}",
            down.shape(),
            c.embedding_size
        );
        Ok(Self {
            gate: None,
            up,
            up_bias: Some(up_bias),
            down,
            down_bias: Some(down_bias),
        })
    }
}
