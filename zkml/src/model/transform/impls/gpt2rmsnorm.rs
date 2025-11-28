//! Definition of the [`RewriteRule`] to replace [`LayerNorm`] with [`RMSNorm`] in a GPT2 model.
use crate::{
    Tensor,
    graph::{Direction, NodeId},
    layers::{
        Layer,
        add::ADD_LAYER,
        einsum::{EINSUM_LAYER, EinSum, axis::Dimension},
        transformer::{
            embeddings::Embeddings,
            layernorm::LayerNorm,
            positional::{POSITIONAL_LAYER, Positional, PositionalVariant, absolute::Absolute},
            rmsnorm::RMSNorm,
        },
    },
    model::{Model, transform::ModelTransform},
    shape::Shape,
    tensor::{KeyedTensor, WrappedTensor},
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
            let Some(Layer::<f32>::LayerNorm(ref layer_norm)) = node.as_inner() else {
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
                old_layer_norm.as_mut().unwrap()
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
            modify_subsequent_linear_layer(output_node, gamma, beta)?;

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
        Layer::<f32>::Positional(positional) => modify_positional(positional),
        Layer::<f32>::Embeddings(embeddings) => modify_embeddings(embeddings),
        Layer::<f32>::EinSum(einsum) => modify_einsum(einsum),
        other => bail!(
            "Expected MatMul, Positional or Embeddings operation, found {}",
            other.short_name()
        ),
    }
}

/// Modify the constant tensors in an [`EinSum`] layer so that the output has rows with mean 0.
fn modify_einsum(einsum: &mut EinSum<f32>) -> Result<()> {
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
            if let Some(tensor) = opt_tensor {
                let new_tensor = mean_subtracted_tensor(&tensor.tensor, dim)?;
                *tensor = tensor.new_map_tensor(|_| new_tensor);
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    einsum.biases.iter_mut().try_for_each(|opt_tensor| {
        if let Some(tensor) = opt_tensor {
            *tensor = tensor.try_new_map_tensor(|bias_tensor| {
                let bias_shape = bias_tensor.shape();
                let bias_sum = bias_tensor.iter().sum::<f32>();
                let bias_mean = bias_sum / bias_shape.numel() as f32;
                let new_bias_data = bias_tensor
                    .iter()
                    .map(|x| x - bias_mean)
                    .collect::<Vec<f32>>();
                Tensor::new(bias_shape.clone(), new_bias_data)
            })?;
        }
        Ok(())
    })
}

/// Modify the embeddings in an [`Embeddings`] layer so that the output has rows with mean 0.
fn modify_embeddings(embeddings: &mut Embeddings<f32>) -> Result<()> {
    // The embedding is just a wrapper around a MatMul with extra info so we call modify_matmul
    embeddings.mat = embeddings.mat.new_map_tensor(mean_subtracted_matrix);
    Ok(())
}

/// Modify the positional encodings in a [`Positional`] layer so that the output has rows with mean 0.
fn modify_positional(positional_layer: &mut Positional<f32>) -> Result<()> {
    // Match on the type of positional encoding, we expect `Learned` here
    *positional_layer = match &positional_layer.variant {
        PositionalVariant::Absolute(absolute) => {
            let Absolute::<f32> { positional, .. } = absolute;
            let new_mat = positional.new_map_tensor(mean_subtracted_matrix);
            Positional::new_absolute(new_mat)
        }
        PositionalVariant::Rope(_) => {
            bail!(
                "Transformation not implemented for Rope, expected to be applicable only with Absolute positional encoding"
            );
        }
    };
    Ok(())
}

/// This function calculates the mean subtraction matrix so that all the output rows have mean 0.
/// It takes as input the final dimension size of the layer.
fn mean_subtracted_matrix(matrix: &Tensor<f32>) -> Tensor<f32> {
    let matrix_shape = matrix.shape();
    let row_size = matrix_shape.dim(-1);

    let subtract_mean_matrix = (0..row_size)
        .flat_map(|i| (0..row_size).map(move |j| if i == j { row_size as f32 - 1.0 } else { -1.0 }))
        .collect::<Vec<f32>>();

    let subtract_mean_tensor =
        Tensor::new(Shape::new(vec![row_size, row_size]), subtract_mean_matrix)
            .expect("Failed to create mean subtraction tensor");
    let mut modified_matrix = matrix
        .matmul(&subtract_mean_tensor)
        .expect("Failed to right-multiply by mean subtraction matrix");
    modified_matrix
        .iter_mut()
        .for_each(|x| *x /= row_size as f32);
    modified_matrix
}

fn mean_subtracted_tensor(tensor: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>> {
    ensure!(
        dim < tensor.rank(),
        "Dimension {dim} out of bounds for tensor of rank {}",
        tensor.rank()
    );
    let shape = tensor.shape();
    let dim_size = shape.dim(dim);

    let subtract_mean_matrix = (0..dim_size)
        .flat_map(|i| {
            (0..dim_size).map(move |j| {
                if i == j {
                    (dim_size as f32 - 1.0) / dim_size as f32
                } else {
                    -1.0 / dim_size as f32
                }
            })
        })
        .collect::<Vec<f32>>();

    let mut input_chars = ('a'..).take(shape.rank()).collect::<Vec<char>>();
    input_chars[dim] = 'm';

    let mut output_chars = input_chars.clone();
    output_chars[dim] = 'n';

    let input_dims = input_chars.iter().collect::<String>();
    let output_dims = output_chars.iter().collect::<String>();

    let equation = format!("I({input_dims})@M(mn)->O({output_dims})");

    let subtract_mean_tensor =
        Tensor::new(Shape::new(vec![dim_size, dim_size]), subtract_mean_matrix)?;

    let einsum = EinSum::<f32>::new(equation, vec![None], vec![None])?;
    let wrapped = WrappedTensor::try_from(tensor)?;
    let wrapped_mean = WrappedTensor::try_from(&subtract_mean_tensor)?;
    let mut output = einsum.evaluate_internal(&[&wrapped, &wrapped_mean])?;

    let out = output.remove(0);
    let data: Vec<f32> = out
        .to_data()
        .to_vec()
        .map_err(|e| anyhow!("Couldn't extract data from tensor: {e:?}"))?;

    Tensor::new(shape.clone(), data)
}

/// This function is used to modify a [`MatMul`] or [`QKV`] layer to absorb the weights and biases from the preceding [`LayerNorm`].
fn modify_subsequent_linear_layer(
    node: &mut Layer<f32>,
    weights: &Tensor<f32>,
    bias: &KeyedTensor<f32>,
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
    bias: &KeyedTensor<f32>,
) -> Result<EinSum<f32>> {
    einsum.reset_caches();

    let wrapped_bias = WrappedTensor::try_from(&bias.tensor())?.unsqueeze_dim(0)?;

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
                let new_bias_data: Vec<f32> = new_bias.to_data().to_vec().map_err(|e| {
                    anyhow!("Couldn't extract data from new einsum bias tensor: {e:?}")
                })?;
                let new_tensor = Tensor::new(shape, new_bias_data)?;
                Ok(Some(old_bias.new_map_tensor(|_| new_tensor)))
            } else {
                // This case only arises when modifying the final projection in GPT2 which has no bias
                // so we can safely squeeze the leading 1 dimension here
                let new_bias = new_bias.squeeze(0)?;
                let shape = new_bias.shape();
                let new_bias_data: Vec<f32> = new_bias.to_data().to_vec().map_err(|e| {
                    anyhow!("Couldn't extract data from new einsum bias tensor: {e:?}")
                })?;
                let new_tensor = Tensor::new(shape.into(), new_bias_data)?;
                Ok(Some(bias.new_map_tensor(|_| new_tensor)))
            }
        })
        .collect::<Result<Vec<Option<KeyedTensor<f32>>>>>()?;

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
            if let Some(tensor) = opt_tensor {
                let new_tensor = tensor.try_new_map_tensor(|t| {
                    let wrapped_t = WrappedTensor::try_from(t)?;
                    let to_cat = wrapped_t
                        .iter_dim(dim)
                        .zip(scales.iter())
                        .map(|(dim_chunk, &scale)| dim_chunk.mul_scalar(scale))
                        .collect::<Vec<WrappedTensor<f32>>>();
                    let cat_tensor = WrappedTensor::cat(to_cat, dim)?;
                    let data: Vec<f32> = cat_tensor.to_data().to_vec().map_err(|e| {
                        anyhow!("Couldn't extract data from tensor during Einsum rescaling: {e:?}")
                    })?;

                    Tensor::new(t.shape().clone(), data)
                })?;
                Ok(Some(new_tensor))
            } else {
                bail!("Need all RHS to be constant tensors in order to rescale Einsum")
            }
        })
        .collect::<Result<Vec<Option<KeyedTensor<f32>>>>>()?;

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
        .map(|t| t.is_some() as usize)
        .sum::<usize>()
        == 3
    {
        "X(se)@WQ(ehd):WK(ehd):WV(ehd)->Q(hsd)+BIAS(hd):K(hsd)+BIAS(hd):V(hsd)+BIAS(hd)".to_string()
    } else if is_final_proj {
        // In this case we also have to change the key on the constant tensor because its no longer the same as the initial embedding matrix
        new_constant_tensors.iter_mut().for_each(|opt_tensor| {
            if let Some(tensor) = opt_tensor {
                tensor.key = "final_proj.weight".into();
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
    use ark_std::rand::Rng;
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
        rng_from_env_or_random,
        tensor::is_close_with_tolerance,
        testing::Pcs,
    };

    use super::*;
    use ff_ext::GoldilocksExt2;

    type F = GoldilocksExt2;

    #[test]
    fn test_mean_subtraction_matrix() {
        let mut rng = rng_from_env_or_random();
        // First we create a random matrix for our "constant right hand side"
        let const_col_size: usize = rng.gen_range(4..20);
        let const_row_size: usize = rng.gen_range(4..20);
        let const_shape = Shape::new(vec![const_col_size, const_row_size]);
        let const_matrix = Tensor::<f32>::random(&const_shape);
        // Make the mean subtraction matrix
        let modified_const = mean_subtracted_matrix(&const_matrix);

        for _ in 0..5 {
            let input_num_cols: usize = rng.gen_range(4..20);
            let input_shape = Shape::new(vec![input_num_cols, const_col_size]);
            let input_matrix = Tensor::<f32>::random(&input_shape);

            let mul_result = input_matrix.matmul(&modified_const).unwrap();
            let result_without_mean = input_matrix.matmul(&const_matrix).unwrap();
            mul_result
                .slice_last_dim()
                .zip(result_without_mean.slice_last_dim())
                .for_each(|(row, row_without_mean)| {
                    let sum = row_without_mean.iter().sum::<f32>();
                    let mean = sum / row.len() as f32;
                    row.iter().zip(row_without_mean.iter()).for_each(|(a, b)| {
                        let diff = a - (b - mean);
                        assert!(diff.abs() < 1e-6, "Difference is too large: {diff}");
                    });
                })
        }
    }

    #[test]
    fn test_mean_subtracted_tensor() -> anyhow::Result<()> {
        let mut rng = rng_from_env_or_random();
        for dim in 0..3 {
            let shape = Shape::new(vec![
                rng.gen_range(4..10),
                rng.gen_range(4..10),
                rng.gen_range(4..10),
            ]);
            let tensor = Tensor::<f32>::random(&shape);
            let modified_tensor = mean_subtracted_tensor(&tensor, dim)?;

            let iter_shape = shape.clone();
            let mut indices = vec![0; shape.rank()];
            let total_iters = shape.numel() / shape.dim(dim);
            for _ in 0..total_iters {
                let mut sum = 0.0;
                for i in 0..shape.dim(dim) {
                    indices[dim] = i;
                    sum += tensor.get(indices.clone())?;
                }
                let mean = sum / shape.dim(dim) as f32;
                for i in 0..shape.dim(dim) {
                    indices[dim] = i;
                    let modified_value = modified_tensor.get(indices.clone())?;
                    let original_value = tensor.get(indices.clone())?;
                    let expected_value = original_value - mean;
                    let diff = modified_value - expected_value;
                    assert!(
                        diff.abs() < 1e-6,
                        "Difference is too large at index {:?}: {diff}",
                        indices
                    );
                }
                // Increment indices for next iteration
                for i in (0..shape.rank()).rev() {
                    if i == dim {
                        continue;
                    }
                    indices[i] += 1;
                    if indices[i] < iter_shape.dim(i) {
                        break;
                    } else {
                        indices[i] = 0;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_gpt2_replace() -> anyhow::Result<()> {
        init_test_logging("debug");
        // First we load up a GPT-2 model
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let gguf = RawGGUF::new(model_path);
        let driver = Driver::load_from_model(GPT2, &gguf, Some(10))?;
        // Extract the model
        let Driver { model, .. } = driver;
        model.describe();
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = GPT2.load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let input_tokens = user_tokens
            .into_iter()
            .map(|t| t.as_number::<f32>())
            .collect::<Vec<f32>>();

        let tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone())?;
        let mut store = GenStore::default();

        let trace = model.run::<F>(vec![tensor.clone()], &mut store)?;
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

        let new_trace = model.run::<F>(vec![tensor.clone()], &mut store)?;

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
        let driver = Driver::load_from_model(GPT2, &gguf, Some(max_context))?;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = GPT2.load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);

        // Rewrite the model by applying our transformation rule
        let conf = default_pipeline_config().with_float_rules(vec![Box::new(GPT2RMSNorm)]);

        let (driver, _metadata) = driver.into_provable_llm(Some(conf))?;
        let trace = driver.run::<GoldilocksExt2>(user_tokens.clone(), &mut GenStore::default())?;

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
