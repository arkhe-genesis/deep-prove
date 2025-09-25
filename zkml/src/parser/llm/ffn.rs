use crate::{
    Number,
    layers::{
        activation::{Activation, GeGlu},
        matrix_mul::MatMul,
        provable::{Edge, Node},
    },
    parser::{
        gguf::FileTensorLoader,
        llm::{LLMVariant, NodeId},
    },
};
use anyhow::{bail, ensure};

use crate::{
    Tensor,
    layers::Layer,
    model::Model,
    parser::{json, llm::LLMConfig},
};

#[derive(Debug, Clone)]
pub struct FeedForward<N: Number> {
    pub gate: Option<MatMul<N>>, // used only for a Gated Linear Unit (GLU)
    pub up: Tensor<N>,
    pub up_bias: Option<Tensor<N>>,
    pub down: Tensor<N>,
    pub down_bias: Option<Tensor<N>>,
}

impl FeedForward<f32> {
    pub fn write_to_model(
        self,
        _config: &LLMConfig,
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
            let geglu = Activation::new_geglu();
            // build input wires for GeGlu
            let geglu_inputs = {
                let mut inputs = vec![Edge::default(); 2];
                inputs[GeGlu::<f32>::GELU_INPUT_INDEX] = Edge::new(gate_node_id, 0); // output of gate is fed to Gelu
                inputs[GeGlu::<f32>::LINEAR_INPUT_INDEX] = Edge::new(up_node_id, 0);
                inputs
            };
            model.add_node(Node::new(geglu_inputs, geglu.into()))
        } else {
            // if there is no `self.gate`, then we just feed output of `up` linear component to the activation layer
            model.add_consecutive_layer(Layer::Activation(Activation::new_gelu()), Some(up_node_id))
        }?;
        let last_node_id =
            model.add_consecutive_layer(Layer::MatMul(down), Some(activation_node_id))?;
        Ok(last_node_id)
    }
    // Replaces from_var_builder and from_tensor_loader
    // 'loader' is expected to be the block-level loader (e.g., scoped to "blk.N.")
    pub fn from_loader(loader: &FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let gate = match &c.variant {
            LLMVariant::GPT2 => None,
            LLMVariant::Gemma3 => {
                let gate = loader.get_tensor("ffn_gate.weight")?.transpose();
                ensure!(
                    gate.shape()[0] == c.hidden_size,
                    "gate have shape {:?} but in features should be equal to hidden_size: {}",
                    gate.shape(),
                    c.hidden_size
                );
                Some(MatMul::new_constant(gate, None)?)
            }
        };

        let up = loader.get_tensor("ffn_up.weight")?.transpose();
        let up_bias = if c.variant.has_biases() {
            Some(loader.get_tensor("ffn_up.bias")?)
        } else {
            None
        };
        let down = loader.get_tensor("ffn_down.weight")?.transpose();
        let down_bias = if !c.variant.has_biases() {
            None
        } else {
            Some(loader.get_tensor("ffn_down.bias")?)
        };
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
            gate,
            up,
            up_bias,
            down,
            down_bias,
        })
    }

    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        if let LLMVariant::Gemma3 = c.variant {
            bail!("Gemma3 is not supported yet for custom JSON format");
        }
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
