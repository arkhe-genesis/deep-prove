//! Model transform for Gemma3 models that absorbs some of the normalisation weights into following layers where possible.

use crate::{
    Tensor,
    graph::Direction,
    layers::{
        Layer,
        einsum::{EinSum, axis::Dimension},
        transformer::normalisation::rmsnorm::RMSNorm,
    },
    model::{Model, transform::ModelTransform},
    tensor::{KeyedTensor, TensorHandle, WrappedTensor},
};
use anyhow::{Result, anyhow, bail, ensure};

#[derive(Debug, Clone, Copy)]
/// Rewrite to absorb RMSNorm weights into following layers where possible, currently this transformation should only be used with Gemma3 Models.
pub struct Gemma3Transform;

impl ModelTransform<f32> for Gemma3Transform {
    fn apply(&self, mut model: Model<f32>) -> Result<Model<f32>> {
        // Iterate over the nodes in `eval_order`
        // NOTE: collecting here because then inside the loop we're mutating the graph
        for node_id in model.eval_order().collect::<Vec<_>>().into_iter() {
            let node = &model.graph[node_id];

            if let Some(Layer::<f32>::RMSNorm(rms_norm)) = node.as_inner() {
                // If we find an RMSNorm already present, we skip it
                let RMSNorm {
                    alpha,
                    eps,
                    normalisation_dim_size,
                    ..
                } = rms_norm.clone();

                // Now we must modify the layers following RMSNorm to absorb the scaling
                // if there is any
                if let Some(ref alpha_inner) = alpha {
                    let mut output_edges = model
                        .graph
                        .neighbors(node_id, Direction::Outgoing)
                        .map(|(_, edge)| edge)
                        .collect::<Vec<_>>();
                    ensure!(
                        output_edges.len() == 1,
                        "Expected RMSNorm to have 1 output, found {}",
                        output_edges.len()
                    );

                    let output_edge = output_edges.remove(0);
                    // there should only be a single edge here as well
                    ensure!(
                        output_edge.ports().len() == 1,
                        "Expected RMSNorm to have 1 port on output edge, found {}",
                        output_edge.ports().len()
                    );

                    // the output edge should have a NodeId so we use that get the next node
                    // safe unwrap since it's guaranteed to be a node because we used node_neighbors
                    let gamma = Tensor::try_from(alpha_inner)?;

                    let output_node_id = output_edge.target();
                    let output_node = model
                        .graph
                        .node_mut(output_node_id)
                        .expect("Output node should exist in the model")
                        .as_inner_mut()
                        .unwrap();

                    let layer_modified = modify_subsequent_layer(output_node, gamma.as_ref())?;
                    if layer_modified {
                        let new_layer = Layer::RMSNorm(RMSNorm::<f32>::new(
                            None,
                            eps,
                            Some(normalisation_dim_size),
                        )?);
                        model.graph.replace_inner(node_id, |_| new_layer)?;
                    } else if matches!(output_node, Layer::<f32>::Positional(..)) {
                        // In this case there is no benefit to absorbing the `alpha` tensor into the following layer so
                        // we simply continue.
                        continue;
                    } else {
                        // In this case we are actually going to insert an EinSum layer that, for the time being,
                        // will perform the scaling operation
                        let identity = Tensor::<f32>::diagonal(gamma.shape().numel(), 1.0f32);
                        let wrapped_identity = WrappedTensor::try_from(&identity)?;
                        let wrapped_gamma = WrappedTensor::try_from(gamma.as_ref())?;
                        let scaling_tensor =
                            wrapped_identity.mul(wrapped_gamma.unsqueeze_dim(0)?)?;
                        let native_scaling_tensor = Tensor::try_from(scaling_tensor)?;
                        let keyed_tensor = KeyedTensor::new(
                            format!("rmsnorm-scaling-{}", node_id),
                            native_scaling_tensor,
                        );
                        let scaling_einsum = EinSum::new(
                            "X(se)@WS(ef)->O(sf)".into(),
                            vec![Some(keyed_tensor.into())],
                            vec![None],
                        )?
                        .no_requant();
                        // here we collect port links from the source port, since we add one EinSum _per source port_ only

                        // we can already delete the outgoing edges from the input node now that we have collected all info necessary
                        // to do the link with EinSum layers
                        let (edge_id, old_edge) = model
                            .graph
                            .neighbors(node_id, Direction::Outgoing)
                            .collect::<Vec<_>>()[0];

                        let edge_id = *edge_id;
                        let ports = old_edge.ports();
                        let target = old_edge.target();
                        let port_link = ports.0[0].clone();
                        let source_port = port_link.source_port;
                        let target_port = port_link.target_port;
                        model.graph.remove_edge(&edge_id)?;

                        // first add the EinSum node to be able to  reference it later
                        // when modifying the edges of the model
                        let einsum_node_id =
                            model.graph.add_inner(Layer::EinSum(scaling_einsum))?;
                        // we create this new port link as the edge from input node ->
                        // EinSum. Given there is only **one** portlink on **one** edge
                        // between input_node_id and this EinSum, we always set
                        // target_port to 0, e.g. first slot.
                        model.graph.add_edge(
                            node_id,
                            einsum_node_id,
                            (source_port.into(), 0),
                            None,
                        )?;
                        model.graph.add_edge(
                            einsum_node_id,
                            target,
                            (0, target_port.into()),
                            None,
                        )?;

                        let new_layer = Layer::RMSNorm(RMSNorm::<f32>::new(
                            None,
                            eps,
                            Some(normalisation_dim_size),
                        )?);
                        model.graph.replace_inner(node_id, |_| new_layer)?;
                    }
                } else {
                    continue;
                }
            }
        }
        Ok(model)
    }
}

/// This function is used to modify an [`EinSum`] layer to absorb the scaling weights from the preceding [`RMSNorm`].
fn modify_subsequent_layer(node: &mut Layer<f32>, weights: &Tensor<f32>) -> Result<bool> {
    match &node {
        Layer::<f32>::EinSum(einsum) => {
            *node = Layer::EinSum(rescale_einsum(einsum, weights)?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn rescale_einsum(einsum: &EinSum<f32>, scales: &Tensor<f32>) -> Result<EinSum<f32>> {
    einsum.reset_caches();

    let rescaled_axis = einsum.mapping.get_lhs_axis_at_dim(-1)?;

    let dims_to_slice_on = rescaled_axis
        .rhs_inputs
        .iter()
        .map(|dim_enum| {
            if let Dimension::Present(dim) = dim_enum {
                Ok(*dim)
            } else {
                Err(anyhow!("Expected RHS dimension to be present in Einsum"))
            }
        })
        .collect::<Result<Vec<usize>>>()?;

    let is_final_proj = einsum.constant_tensors.iter().any(|t| {
        if let Some(tensor) = t {
            // Gemma3 vocab size is 262144 so we can use this to identify the final projection
            tensor.shape().dim(0) == 262144
        } else {
            false
        }
    });

    let mut new_constant_tensors = einsum
        .constant_tensors
        .iter()
        .zip(dims_to_slice_on)
        .map(|(opt_tensor, dim)| {
            if let Some(tensor) = opt_tensor {
                let wrapped_t = tensor.wrapped_tensor()?.clone();

                let to_cat = wrapped_t
                    .iter_dim(dim)
                    .zip(scales.data().iter())
                    .map(|(dim_chunk, &scale)| dim_chunk.mul_scalar(scale))
                    .collect::<Vec<WrappedTensor<f32>>>();
                let cat_tensor = WrappedTensor::cat(to_cat, dim)?;

                Ok(Some(TensorHandle::from_wrapped_tensor(
                    tensor.storage_key().to_owned(),
                    tensor.store().clone(),
                    cat_tensor,
                )))
            } else {
                bail!("Need all RHS to be constant tensors in order to rescale Einsum")
            }
        })
        .collect::<Result<Vec<Option<TensorHandle<f32>>>>>()?;

    if is_final_proj {
        // In this case we also have to change the key on the constant tensor because its no longer the same as the initial embedding matrix
        new_constant_tensors.iter_mut().for_each(|opt_tensor| {
            if let Some(tensor) = opt_tensor {
                tensor.set_storage_key("final_proj.weight".into());
            }
        });
    }

    let equation = einsum.equation.clone();
    let concatenation_dims = einsum
        .caches
        .iter()
        .map(|c| c.clone().map(|cache| cache.lock().unwrap().cache_info().1))
        .collect::<Vec<Option<usize>>>();

    let mut output = EinSum::new(equation, new_constant_tensors, einsum.biases.clone())?;
    output.with_caches(concatenation_dims)?;
    if let Some(name) = einsum.name.as_ref() {
        output = output.with_name(name.clone());
    }
    if !einsum.requantise {
        output = output.no_requant();
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use tenstore::GenStore;

    use crate::{
        graph::NodeId,
        init_test_logging,
        model::llm::Driver,
        parser::{
            llm::{
                LLMTokenizer,
                models::gemma3::{Gemma3, safe_tests::GEMMA3_SAFE_MODEL},
                tokenizer::TokenizerLoader,
            },
            safe::RawSafeTensors,
        },
        tensor::is_close_with_tolerance,
    };

    use super::*;

    #[test]
    fn test_gemma3_replace() -> anyhow::Result<()> {
        init_test_logging("debug");
        // First we load up a Gemma3 model
        let raw = RawSafeTensors::from_hugging_face_cached(GEMMA3_SAFE_MODEL)?;

        let gemma3 = Gemma3::new();
        let driver = Driver::load_from_model(gemma3, &raw, Some(10))?;
        // Extract the model
        let Driver { model, .. } = driver;
        model.describe();
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = gemma3.load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let input_tokens = user_tokens
            .into_iter()
            .map(|t| t.as_number::<f32>())
            .collect::<Vec<f32>>();

        let tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone())?;
        let mut store = GenStore::default();

        let trace = model.run(vec![tensor.clone()], &mut store)?;
        // Get the final node of the Model, we will compare the inputs to this node before and after the transformation (we compare the inputs because the outputs of this layer are tokens
        // and it may be the case that we would get the same tokens out but the actual logits are different)
        let last_model_node_id = model
            .graph
            .backward_iter()
            .filter(|(_, n)| n.is_inner())
            .take(1)
            .map(|(id, _)| id)
            .collect::<Vec<NodeId>>()[0];
        // Extract the input to the Logits layer before applying the transformation.
        let pre_transform_final_step = trace.get_step(&last_model_node_id).unwrap();
        let pre_transform_inputs = pre_transform_final_step.input_tensors().unwrap();
        // Rewrite the model by applying our transformation rule
        let model = Gemma3Transform.apply(model)?;
        model.reset();
        // Now we generate the post-transformation trace and extract the logits step data
        let mut store = GenStore::default();

        let new_trace = model.run(vec![tensor.clone()], &mut store)?;

        let post_transform_final_step = new_trace.get_step(&last_model_node_id).unwrap();
        let post_transform_inputs = post_transform_final_step.input_tensors().unwrap();
        // Compare the pre and post transformation data
        for (pre, post) in pre_transform_inputs
            .iter()
            .zip(post_transform_inputs.iter())
        {
            assert_eq!(pre.shape(), post.shape());
            let pre_data = pre.data();
            let post_data = post.data();
            // for (i, (pre, post)) in pre_data.iter().zip(post_data.iter()).enumerate() {
            //     let diff = (pre - post).abs();
            //     if diff > 1e-4 {
            //         println!(
            //             "Transformed Model output was not close to the original at index {}: pre value {}, post value {}, diff {}",
            //             i, pre, post, diff
            //         );
            //     }
            // }
            assert!(
                is_close_with_tolerance(pre_data, post_data, 1e-4, 1e-4),
                "Transformed Model output was not close to the original, first 10 pre data: {:?}, first 10 post data: {:?}",
                &pre_data[..10],
                &post_data[..10],
            );
        }
        Ok(())
    }
}
