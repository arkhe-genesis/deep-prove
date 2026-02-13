//! Definition of the [`RewriteRule`] to replace [`LayerNorm`] with [`RMSNorm`] in a GPT2 model.
use crate::{
    Tensor,
    graph::{Direction, NodeId},
    layers::{
        Layer,
        add::ADD_LAYER,
        einsum::{EINSUM_LAYER, EinSum, axis::Dimension},
        transformer::{
            layernorm::LayerNorm,
            positional::{POSITIONAL_LAYER, Positional, PositionalVariant, absolute::Absolute},
            rmsnorm::RMSNorm,
        },
    },
    model::{Model, transform::ModelTransform},
    shape::Shape,
    tensor::{TensorHandle, WrappedTensor},
};
use anyhow::{Result, anyhow, bail, ensure};

#[derive(Debug)]
/// Rewrite rule to replace LayerNorm with RMSNorm, currently this transformation should only be used with GPT2 Models.
pub struct GPT2RMSNorm;

impl ModelTransform<f32> for GPT2RMSNorm {
    fn apply(&self, mut model: Model<f32>) -> Result<Model<f32>> {
        // Iterate over the nodes in `eval_order`
        // NOTE: collecting here because then inside the loop we're mutating the graph
        for node_id in model.eval_order().collect::<Vec<_>>().into_iter() {
            let node = &model.graph[node_id];

            // If the node isn't a LayerNorm, do nothing
            let Some(Layer::<f32>::LayerNorm(layer_norm)) = node.as_inner() else {
                continue;
            };

            let LayerNorm { gamma, eps, .. } = layer_norm;

            let mut old_layer_norm = None;
            // Create the new RMSNorm node
            let rms_norm = Layer::RMSNorm(RMSNorm::<f32>::new(None, *eps, Some(gamma.shape()[0]))?);

            // at this point we dont need the immutable reference - everything
            // below is mutable so we take back ownership of the layer norm
            // without copying by swapping it with the new RMSNorm and then
            // later can change the associated nodes
            model.graph.replace_inner(node_id, |ln| {
                old_layer_norm = Some(ln);
                rms_norm
            })?;

            let Layer::<f32>::LayerNorm(LayerNorm { gamma, beta, .. }) =
                old_layer_norm.as_ref().unwrap()
            else {
                unreachable!("Expected LayerNorm node");
            };

            // Now we must modify the layers following LayerNorm
            // We should have a single output to the LayerNorm so we check that here
            let mut output_edges = model
                .graph
                .neighbors(node_id, Direction::Outgoing)
                .map(|(_, edge)| edge)
                .collect::<Vec<_>>();
            ensure!(
                output_edges.len() == 1,
                "Expected LayerNorm to have 1 output, found {}",
                output_edges.len()
            );

            let output_edge = output_edges.remove(0);
            // there should only be a single edge here as well
            ensure!(
                output_edge.ports().len() == 1,
                "Expected LayerNorm to have 1 output edge, found {}",
                output_edge.ports().len()
            );

            // the output edge should have a NodeId so we use that get the next node
            // safe unwrap since it's guaranteed to be a node because we used node_neighbors
            #[allow(clippy::clone_on_copy)]
            let output_node_id = output_edge.target().clone();
            let output_node = model
                .graph
                .node_mut(output_node_id)
                .expect("Output node should exist in the model")
                .as_inner_mut()
                .unwrap();
            modify_subsequent_linear_layer(output_node, &Tensor::try_from(gamma)?, beta)?;

            let input_node_ids = model
                .graph
                .incomings(node_id)
                .map(|(_, edge)| edge.source())
                .filter(|n_id| model.graph.node(*n_id).unwrap().as_inner().is_some())
                .collect::<Vec<_>>();

            // Modify the input and output nodes as required
            for input_node_id in input_node_ids.into_iter() {
                let Some(input_node) = model.graph[input_node_id].as_inner() else {
                    unreachable!("filtered above")
                };

                let input_op_name = input_node.short_name();
                match input_op_name {
                    ADD_LAYER => {
                        add_was_previous_layer(&mut model, input_node_id)?;
                    }
                    POSITIONAL_LAYER => {
                        positional_was_previous_layer(&mut model, input_node_id)?;
                    }
                    _ => bail!("Unexpected layer type: {input_op_name}"),
                }
            }
        }
        Ok(model)
    }
}

/// Function used when the layer prior to [`LayerNorm`] was an Add. Checks the inputs of the Add to ensure they are
/// either Add, [`Positional`] or [`EinSum`] and that there is at least one [`EinSum`].
fn add_was_previous_layer(model: &mut Model<f32>, input_node_id: NodeId) -> Result<()> {
    let input_ids = model
        .graph
        .incomings(input_node_id)
        .map(|(_, edge)| edge.source())
        .filter(|n_id| model.graph[*n_id].as_inner().is_some())
        .collect::<Vec<_>>();

    let seen_an_einsum = input_ids
        .into_iter()
        .try_fold(false, |seen_an_einsum, input_id| {
            // safe unwrap since it's guaranteed to be a node because we used node_neighbors
            let Some(input_node) = model.graph.node_mut(input_id).unwrap().as_inner_mut() else {
                unreachable!("filtered above")
            };
            let add_input_op_name = input_node.short_name();
            match add_input_op_name {
                EINSUM_LAYER => {
                    // Found a LayerNorm node with an Add layer as input, which has a linear layer as input
                    modify_matrix_subtract_mean(input_node)?;
                    Ok(true)
                }
                ADD_LAYER | POSITIONAL_LAYER => Ok(seen_an_einsum),
                _ => bail!("Expected MatMul or Add layer, found {add_input_op_name}"),
            }
        })?;
    ensure!(
        seen_an_einsum,
        "Expected to find a Einsum layer as input to the Add {} layer before LayerNorm, found none",
        input_node_id
    );
    Ok(())
}

/// Function used when the layer prior to [`LayerNorm`] was a [`Positional`]. Checks the [`Positional`] has a singular input
/// which is an [`Embeddings`]. Then it modifies both the [`Positional`] and [`Embeddings`] layers so each row has mean 0.
fn positional_was_previous_layer(model: &mut Model<f32>, input_node_id: NodeId) -> Result<()> {
    let positional_node = model
        .graph
        .node_mut(input_node_id)
        .expect("Input node should exist in the model")
        .as_inner_mut()
        .unwrap();
    // If we have a positional layer we have to modify it and the preceding embeddings layer
    modify_matrix_subtract_mean(positional_node)?;

    let positional_inputs = model
        .graph
        .neighbors(input_node_id, Direction::Incoming)
        .collect::<Vec<_>>();
    // We check that this has length 1
    ensure!(
        positional_inputs.len() == 1,
        "Expected positional layer to have 1 input"
    );
    // Now we need to modify the preceding embeddings layer
    // safe unwrap since it's guaranteed to be a node because we used node_neighbors
    let embeddings_node_id = positional_inputs[0].1.source();
    let embeddings_node = model
        .graph
        .node_mut(embeddings_node_id)
        .expect("Embeddings node should exist in the model")
        .as_inner_mut()
        .unwrap();
    modify_matrix_subtract_mean(embeddings_node)?;
    Ok(())
}

/// This function is used to modify a [`MatMul`], [`Positional`] or [`Embeddings`] layer so that the rows of the output
/// of the layer will always have mean 0. This is done by right multiplying their respective matrices by a "mean subtraction" matrix,
/// a square matrix with `(row_size - 1) / row_size` along the diagonal and `-1 / row_size` everywhere else.
fn modify_matrix_subtract_mean(node: &mut Layer<f32>) -> Result<()> {
    match node {
        Layer::<f32>::Positional(positional) => {
            *positional = match &mut positional.variant {
                PositionalVariant::Absolute(absolute) => {
                    let Absolute::<f32> { positional, .. } = absolute;
                    let wrapped_tensor = positional.take_wrapped_tensor()?;
                    let centered = wrapped_tensor.mean_center_rows();
                    let mut handle = positional.clone();
                    handle.set_wrapped_tensor(centered)?;
                    Positional::new_absolute(handle)?
                }
                PositionalVariant::Rope(_) => {
                    bail!(
                        "Transformation not implemented for Rope, expected to be applicable only with Absolute positional encoding"
                    );
                }
            };
            Ok(())
        }
        Layer::<f32>::Embeddings(embeddings) => {
            // temporarily take ownership of the tensor, this allows for modify in place
            let tensor = embeddings.mat.take_wrapped_tensor()?;
            let centered = tensor.mean_center_rows();
            embeddings.mat.set_wrapped_tensor(centered)?;
            Ok(())
        }
        Layer::<f32>::EinSum(einsum) => {
            let axes = einsum.mapping.get_final_output_axes();

            let dims_to_modify = axes
                .iter()
                .enumerate()
                .map(|(i, axis)| {
                    if let Dimension::Present(rhs_dim) = axis.rhs_inputs[i] {
                        Ok(rhs_dim)
                    } else {
                        Err(anyhow!("Expected RHS dimension to be present in Einsum"))
                    }
                })
                .collect::<Result<Vec<usize>>>()?;

            einsum
                .constant_tensors
                .iter_mut()
                .zip(dims_to_modify)
                .try_for_each(|(opt_tensor, dim)| {
                    if let Some(mut handle) = opt_tensor.take() {
                        let wrapped = handle.take_wrapped_tensor()?;
                        let centered = wrapped.mean_center_dim(dim);
                        handle.set_wrapped_tensor(centered)?;
                        *opt_tensor = Some(handle);
                        Ok(())
                    } else {
                        bail!("Expected constant tensor in Einsum to modify")
                    }
                })?;

            einsum.biases.iter_mut().try_for_each(|opt_tensor| {
                if let Some(handle) = opt_tensor {
                    let wrapped = handle.take_wrapped_tensor()?;
                    let shape = wrapped.shape();
                    let flatten = wrapped.flatten_1d();
                    let centered = flatten.mean_center_rows();
                    let result = centered.reshape(shape)?;
                    handle.set_wrapped_tensor(result)?;
                }
                Ok(())
            })
        }
        other => bail!(
            "Expected MatMul, Positional or Embeddings operation, found {}",
            other.short_name()
        ),
    }
}

/// This function is used to modify a [`MatMul`] or [`QKV`] layer to absorb the weights and biases from the preceding [`LayerNorm`].
fn modify_subsequent_linear_layer(
    node: &mut Layer<f32>,
    weights: &Tensor<f32>,
    bias: &TensorHandle<f32>,
) -> Result<()> {
    *node = match &node {
        Layer::<f32>::EinSum(einsum) => Layer::EinSum(rescale_einsum(einsum, weights, bias)?),
        other => bail!(
            "Expected MatMul or QKV operation, found {}",
            other.short_name()
        ),
    };
    Ok(())
}

fn rescale_einsum(
    einsum: &EinSum<f32>,
    scales: &Tensor<f32>,
    bias: &TensorHandle<f32>,
) -> Result<EinSum<f32>> {
    einsum.reset_caches();

    let wrapped_bias = bias.wrapped_tensor()?.clone().unsqueeze_dim(0)?;

    let new_biases = einsum.evaluate_internal(&[&wrapped_bias])?;
    let biases = einsum
        .biases
        .iter()
        .zip(new_biases.into_iter())
        .map(|(old_bias, new_bias)| {
            if let Some(old_bias) = old_bias {
                let shape: Shape = old_bias
                    .shape()
                    .iter()
                    .filter_map(|&d| if d != 1 { Some(d) } else { None })
                    .collect::<Vec<usize>>()
                    .into();
                let reshaped = new_bias.reshape(shape.into())?;
                let mut handle = old_bias.clone();
                handle.set_wrapped_tensor(reshaped)?;
                Ok(Some(handle))
            } else {
                // This case only arises when modifying the final projection in GPT2 which has no bias
                // so we can safely squeeze the leading 1 dimension here
                let new_bias = new_bias.squeeze(0)?;
                let mut handle = bias.clone();
                handle.set_wrapped_tensor(new_bias)?;
                Ok(Some(handle))
            }
        })
        .collect::<Result<Vec<Option<_>>>>()?;

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

    let mut new_constant_tensors = einsum
        .constant_tensors
        .iter()
        .zip(dims_to_slice_on)
        .map(|(opt_tensor, dim)| {
            if let Some(handle) = opt_tensor {
                let wrapped = handle.take_wrapped_tensor()?;
                // TODO: make a single tensor
                let to_cat = wrapped
                    .iter_dim(dim)
                    .zip(scales.data())
                    .map(|(dim_chunk, &scale)| dim_chunk.mul_scalar(scale))
                    .collect::<Vec<WrappedTensor<f32>>>();
                let cat_tensor = WrappedTensor::cat(to_cat, dim)?;
                let mut handle = handle.clone();
                handle.set_wrapped_tensor(cat_tensor)?;
                Ok(Some(handle))
            } else {
                bail!("Need all RHS to be constant tensors in order to rescale Einsum")
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let is_final_proj = einsum.constant_tensors.iter().any(|t| {
        if let Some(tensor) = t {
            // GPT2 vocab size is 50257 so we can use this to identify the final projection
            tensor.shape().dim(0) == 50257
        } else {
            false
        }
    });
    let equation = if einsum
        .constant_tensors
        .iter()
        .map(|handle_opt| handle_opt.is_some() as usize)
        .sum::<usize>()
        == 3
    {
        "X(se)@WQ(ehd):WK(ehd):WV(ehd)->Q(hsd)+BIAS(hd):K(hsd)+BIAS(hd):V(hsd)+BIAS(hd)".to_string()
    } else if is_final_proj {
        // In this case we also have to change the key on the constant tensor because its no longer the same as the initial embedding matrix
        new_constant_tensors.iter_mut().for_each(|tensor_opt| {
            if let Some(tensor) = tensor_opt {
                tensor.set_storage_key("final_proj.weight".into());
            }
        });
        "X(se)@WE(ve)->O(sv)+BIAS(v)".to_string()
    } else {
        "X(se)@WU(ep)->U(sp)+BIAS(p)".to_string()
    };
    let concatenation_dims = einsum
        .caches
        .iter()
        .map(|c| c.clone().map(|cache| cache.lock().unwrap().cache_info().1))
        .collect::<Vec<Option<usize>>>();

    let mut output = EinSum::new(equation, new_constant_tensors, biases)?;
    output.with_caches(concatenation_dims)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use tenstore::GenStore;

    use crate::{
        init_test_logging,
        model::llm::{Driver, WithMaxContext},
        parser::{
            default_pipeline_config, file_cache,
            gguf::RawGGUF,
            llm::{
                LLMTokenizer,
                models::gpt2::{GPT2, GPT2_Q8_0},
                tokenizer::TokenizerLoader,
            },
        },
        tensor::is_close_with_tolerance,
        testing::Pcs,
    };

    use super::*;
    use ff_ext::GoldilocksExt2;

    #[test]
    fn test_gpt2_replace() -> anyhow::Result<()> {
        init_test_logging("debug");
        // First we load up a GPT-2 model
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let gguf = RawGGUF::new(model_path);
        let driver = Driver::load_from_model(GPT2::new(), &gguf, Some(10))?;
        // Extract the model
        let Driver { model, .. } = driver;
        model.describe();
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = GPT2::new().load_tokenizer(&gguf)?;
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
        let model = GPT2RMSNorm.apply(model)?;

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
            assert!(
                is_close_with_tolerance(pre_data, post_data, 0.0, 1e-4),
                "Transformed Model output was not close to the original, first 10 pre data: {:?}, first 10 post data: {:?}",
                &pre_data[..10],
                &post_data[..10],
            );
        }
        Ok(())
    }

    #[test]
    fn test_gpt2_replace_proving() -> Result<()> {
        init_test_logging("debug");
        let max_context = 10;
        // First we load up a GPT-2 model
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let gguf = RawGGUF::new(model_path);
        let driver = Driver::load_from_model(GPT2::new(), &gguf, Some(max_context))?;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = GPT2::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);

        // Rewrite the model by applying our transformation rule
        let conf = default_pipeline_config().with_float_rules(vec![Box::new(GPT2RMSNorm)]);

        let (driver, _metadata) = driver.into_provable_llm(Some(conf))?;
        let input_tensors = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensors, &mut GenStore::default())?;

        let (prover_ctx, verifier_ctx) = driver
            .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
            .with_max_context(max_context);
        let io = trace.to_verifier_io()?;
        let proof = driver.prove(&prover_ctx, trace)?;
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        tracing::info!("Proof size: {}", proof_bytes.len());
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }
}
