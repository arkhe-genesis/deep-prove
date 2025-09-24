//! Definition of the [`RewriteRule`] to replace [`LayerNorm`] with [`RMSNorm`]
use crate::{
    Tensor,
    layers::{
        Layer,
        add::ADD_LAYER,
        matrix_mul::{Config, MATMUL_LAYER, MatMul, OperandMatrix},
        provable::{Node, NodeId},
        transformer::{
            embeddings::Embeddings,
            layernorm::{LAYERNORM_LAYER, LayerNorm},
            positional::{POSITIONAL_LAYER, Positional, PositionalVariant, absolute::Absolute},
            qkv::QKV,
            rmsnorm::RMSNorm,
        },
    },
    model::{Model, transform::ModelTransform},
    shape::Shape,
};
use anyhow::{Result, anyhow, bail, ensure};

#[derive(Debug)]
/// Rewrite rule to replace LayerNorm with RMSNorm, currently this transformation should only be used with GPT2 Models.
pub struct LayerNormToRMSNorm;

impl ModelTransform<f32> for LayerNormToRMSNorm {
    fn apply(&self, mut model: Model<f32>) -> Result<Model<f32>> {
        let eval_order = model.eval_order();

        // Iterate over the nodes in `eval_order`
        for node_id in eval_order {
            let node_short_name = model
                .nodes
                .get(&node_id)
                .expect("Node should exist")
                .operation
                .short_name();

            // If the node isn't a LayerNorm, do nothing
            if node_short_name != LAYERNORM_LAYER {
                continue;
            }

            // Apply the transformation
            let Node {
                inputs,
                outputs,
                operation,
            } = model
                .nodes
                .remove(&node_id)
                .expect("Already checked LayerNorm exists");
            let Layer::<f32>::LayerNorm(layer_norm) = operation else {
                unreachable!("Already checked this to be LayerNorm operation")
            };
            let LayerNorm {
                gamma, beta, eps, ..
            } = layer_norm;

            // Create the new RMSNorm node
            let rms_norm = RMSNorm::<f32>::new(None, eps, Some(gamma.shape()[0]))?;

            // Modify the input and output nodes as required
            for input_edge in inputs.iter() {
                let input_node_id_opt = &input_edge.node;
                if let Some(input_node_id) = input_node_id_opt {
                    let input_node = model
                        .nodes
                        .get(input_node_id)
                        .expect("Input node should exist in the model");
                    let input_op_name = input_node.operation.short_name();

                    match input_op_name {
                        ADD_LAYER => {
                            add_was_previous_layer(&mut model, *input_node_id)?;
                        }
                        POSITIONAL_LAYER => {
                            positional_was_previous_layer(&mut model, *input_node_id)?;
                        }
                        layer_name => bail!("Unexpected layer type: {layer_name}"),
                    }
                } else {
                    // LayerNorm should have an input node so we return an error if it doesn't
                    bail!("Expected input node for LayerNorm, found None");
                }
            }

            // Now we must modify the layers following LayerNorm
            // We should have a single output to the LayerNorm so we check that here
            ensure!(
                outputs.len() == 1,
                "Expected LayerNorm to have 1 output, found {}",
                outputs.len()
            );

            let output_wire = &outputs[0];
            // there should only be a single edge here as well
            ensure!(
                output_wire.edges.len() == 1,
                "Expected LayerNorm to have 1 output edge, found {}",
                output_wire.edges.len()
            );

            let output_edge = &output_wire.edges[0];
            // the output edge should have a NodeId so we use that get the next node
            let output_node_id_opt = &output_edge.node;
            if let Some(output_node_id) = output_node_id_opt {
                let output_node = model
                    .nodes
                    .get_mut(output_node_id)
                    .expect("Output node should exist in the model");
                *output_node = modify_subsequent_linear_layer(output_node, &gamma, &beta)?;
            } else {
                // The output edge should always have a node
                bail!("Expected output node for LayerNorm, found None");
            }

            // Now insert the RMSNorm that replaces the LayerNorm
            let rms_node = Node {
                inputs,
                outputs,
                operation: Layer::RMSNorm(rms_norm),
            };

            model.nodes.insert(node_id, rms_node);
        }

        Ok(model)
    }
}

/// Function used when the layer prior to [`LayerNorm`] was an Add. Checks the inputs of the Add to ensure they are
/// either Add, [`Positional`] or [`MatMul`] and that there is at least one [`MatMul`].
fn add_was_previous_layer(model: &mut Model<f32>, input_node_id: NodeId) -> Result<()> {
    let input_id_opts = model
        .nodes
        .get(&input_node_id)
        .expect("Input node should exist in the model")
        .inputs
        .iter()
        .map(|edge| edge.node)
        .collect::<Vec<Option<NodeId>>>();

    let mut seen_a_matmul = false;
    input_id_opts.into_iter().try_for_each(|opt_id| {
        match opt_id {
            Some(input_node_id) => {
                let add_input_node = model
                    .nodes
                    .get_mut(&input_node_id)
                    .expect("Add input node should exist in the model");

                let add_input_op_name = add_input_node.operation.short_name();
                match add_input_op_name {
                    MATMUL_LAYER => {
                        // Found a LayerNorm node with an Add layer as input, which has a linear layer as input
                        *add_input_node = modify_matrix_subtract_mean(add_input_node).unwrap();
                        seen_a_matmul = true;
                        Ok(())
                    }
                    ADD_LAYER | POSITIONAL_LAYER => Ok(()),

                    _ => bail!("Expected MatMul or Add layer, found {add_input_op_name}"),
                }
            }
            None => bail!("Expected input node for Add layer, found None"),
        }
    })?;

    if !seen_a_matmul {
        bail!(
            "Expected to find a MatMul layer as input to the Add layer before LayerNorm, found none"
        );
    }
    Ok(())
}

/// Function used when the layer prior to [`LayerNorm`] was a [`Positional`]. Checks the [`Positional`] has a singular input
/// which is an [`Embeddings`]. Then it modifies both the [`Positional`] and [`Embeddings`] layers so each row has mean 0.
fn positional_was_previous_layer(model: &mut Model<f32>, input_node_id: NodeId) -> Result<()> {
    let positional_node = model
        .nodes
        .get_mut(&input_node_id)
        .expect("Input node should exist in the model");
    // If we have a positional layer we have to modify it and the preceding embeddings layer
    *positional_node = modify_matrix_subtract_mean(positional_node)?;

    // Now we need to modify the preceding embeddings layer
    let positional_inputs = &positional_node.inputs;
    // We check that this has length 1
    ensure!(
        positional_inputs.len() == 1,
        "Expected positional layer to have 1 input"
    );
    // Now we need to modify the preceding embeddings layer
    let embeddings_node_id_opt = positional_inputs[0].node;
    if let Some(embeddings_node_id) = embeddings_node_id_opt {
        let embeddings_node = model
            .nodes
            .get_mut(&embeddings_node_id)
            .expect("Embeddings node should exist in the model");
        *embeddings_node = modify_matrix_subtract_mean(embeddings_node)?;
    } else {
        // The positional layer should always have an input node
        bail!("Expected input node for positional layer, found None");
    }
    Ok(())
}

/// This function is used to modify a [`MatMul`], [`Positional`] or [`Embeddings`] layer so that the rows of the output
/// of the layer will always have mean 0. This is done by right multiplying their respective matrices by a "mean subtraction" matrix,
/// a square matrix with `(row_size - 1) / row_size` along the diagonal and `-1 / row_size` everywhere else.
fn modify_matrix_subtract_mean(node: &Node<f32>) -> Result<Node<f32>> {
    match &node.operation {
        Layer::<f32>::MatMul(mat_mul) => modify_matmul(mat_mul).map(|new_mat_mul| Node {
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            operation: Layer::MatMul(new_mat_mul),
        }),
        Layer::<f32>::Positional(positional) => {
            modify_positional(positional).map(|new_positional| Node {
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                operation: Layer::Positional(new_positional),
            })
        }
        Layer::<f32>::Embeddings(embeddings) => {
            modify_embeddings(embeddings).map(|new_embeddings| Node {
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                operation: Layer::Embeddings(new_embeddings),
            })
        }
        other => bail!(
            "Expected MatMul, Positional or Embeddings operation, found {}",
            other.short_name()
        ),
    }
}

/// Modify the constant matrix in a [`MatMul`] layer so that the output has rows with mean 0.
fn modify_matmul(mat_mul: &MatMul<f32>) -> Result<MatMul<f32>> {
    match (&mat_mul.left_matrix, &mat_mul.right_matrix) {
        (OperandMatrix::Weight(_), OperandMatrix::Weight(_)) => Err(anyhow!(
            "Found layer with 2 constant matrices, which is useless as the
                product can be directly used instead"
        )),
        (OperandMatrix::Weight(..), OperandMatrix::Input) => Err(anyhow!(
            "Found MatMul with constant left matrix, this is not supported"
        )),
        (OperandMatrix::Input, OperandMatrix::Weight(mat)) => {
            let new_mat = if let Some(Config::TransposeB) = mat_mul.config {
                mean_subtracted_matrix(&mat.tensor.transpose())
            } else {
                mean_subtracted_matrix(&mat.tensor)
            };

            let weight_matrix = OperandMatrix::new_weight_matrix(new_mat);
            // Now we subtract the bias mean from each element of the bias
            let new_bias = mat_mul.bias.as_ref().map(|old_bias| {
                let bias_shape = old_bias.shape();
                let bias_sum = old_bias.iter().sum::<f32>();
                let bias_mean = bias_sum / bias_shape.dim(0) as f32;
                let new_bias_data = old_bias.iter().map(|x| x - bias_mean).collect::<Vec<f32>>();
                Tensor::new(bias_shape.clone(), new_bias_data)
            });

            // No config now because we have transposed the matrix,
            MatMul::new_internal(OperandMatrix::Input, weight_matrix, new_bias, None)
        }
        (OperandMatrix::Input, OperandMatrix::Input) => Err(anyhow::anyhow!(
            "Found MatMul with 2 input matrices, this is not supported"
        )),
    }
}
/// Modify the embeddings in an [`Embeddings`] layer so that the output has rows with mean 0.
fn modify_embeddings(embeddings: &Embeddings<f32>) -> Result<Embeddings<f32>> {
    // The embedding is just a wrapper around a MatMul with extra info so we call modify_matmul
    let modified_matmul = modify_matmul(&embeddings.mat)?;
    Ok(Embeddings {
        mat: modified_matmul,
        ..embeddings.clone()
    })
}

/// Modify the positional encodings in a [`Positional`] layer so that the output has rows with mean 0.
fn modify_positional(positional_layer: &Positional<f32>) -> Result<Positional<f32>> {
    // Match on the type of positional encoding, we expect `Learned` here
    match &positional_layer.variant {
        PositionalVariant::Absolute(absolute) => {
            let Absolute::<f32> { positional, .. } = absolute;
            let new_mat = mean_subtracted_matrix(positional);
            Ok(Positional::new_absolute(new_mat))
        }
        PositionalVariant::Rope(_) => unimplemented!(
            "Transformation not implemented for Rope, expected to be applicable only with Absolute positional encoding"
        ),
    }
}

/// This function calculates the mean subtraction matrix so that all the output rows have mean 0.
/// It takes as input the final dimension size of the layer.
fn mean_subtracted_matrix(matrix: &Tensor<f32>) -> Tensor<f32> {
    let matrix_shape = matrix.shape();
    let row_size = matrix_shape.dim(matrix_shape.rank() - 1);

    let subtract_mean_matrix = (0..row_size)
        .flat_map(|i| (0..row_size).map(move |j| if i == j { row_size as f32 - 1.0 } else { -1.0 }))
        .collect::<Vec<f32>>();

    let subtract_mean_tensor =
        Tensor::new(Shape::new(vec![row_size, row_size]), subtract_mean_matrix);
    let mut modified_matrix = matrix.matmul(&subtract_mean_tensor);
    modified_matrix
        .iter_mut()
        .for_each(|x| *x /= row_size as f32);
    modified_matrix
}

/// This function is used to modify a [`MatMul`] or [`QKV`] layer to absorb the weights and biases from the preceding [`LayerNorm`].
fn modify_subsequent_linear_layer(
    node: &Node<f32>,
    weights: &Tensor<f32>,
    bias: &Tensor<f32>,
) -> Result<Node<f32>> {
    match &node.operation {
        Layer::<f32>::MatMul(mat_mul) => {
            rescale_matmul(mat_mul, weights, bias).map(|new_mat_mul| Node {
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
                operation: Layer::MatMul(new_mat_mul),
            })
        }
        Layer::<f32>::QKV(qkv) => rescale_qkv_layer(qkv, weights, bias).map(|new_qkv| Node {
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            operation: Layer::QKV(new_qkv),
        }),

        other => Err(anyhow!(
            "Expected MatMul or QKV operation, found {}",
            other.short_name()
        )),
    }
}

/// Function that rescales the weight matrix and modifies the bias of a [`MatMul`] layer.
fn rescale_matmul(
    mat_mul: &MatMul<f32>,
    scales: &Tensor<f32>,
    bias: &Tensor<f32>,
) -> Result<MatMul<f32>> {
    // The matmul should have the right matrix as the constant one
    match (&mat_mul.left_matrix, &mat_mul.right_matrix) {
        (OperandMatrix::Weight(_), OperandMatrix::Weight(_)) => Err(anyhow!(
            "Found layer with 2 constant matrices, which is useless as the
                product can be directly used instead"
        )),
        (OperandMatrix::Weight(..), OperandMatrix::Input) => Err(anyhow!(
            "Found MatMul with constant left matrix, this is not supported"
        )),
        (OperandMatrix::Input, OperandMatrix::Weight(mat)) => {
            let inner_mat = if let Some(Config::TransposeB) = mat_mul.config {
                mat.tensor.transpose()
            } else {
                mat.tensor.clone()
            };
            // We transform the bias so it is a `1 x bias_size` matrix
            let mut matrix_bias = bias.clone();
            matrix_bias.reshape(Shape::new(vec![1, bias.shape().dim(0)]));
            let new_bias_shape = Shape::new(vec![inner_mat.ncols_2d()]);
            let mut new_bias = matrix_bias.matmul(&inner_mat);
            new_bias.reshape(new_bias_shape);

            let new_mat_data = inner_mat
                .slice_last_dim()
                .zip(scales.iter())
                .flat_map(|(row, scale)| row.iter().map(|x| x * scale).collect::<Vec<f32>>())
                .collect::<Vec<f32>>();
            let new_mat = Tensor::new(inner_mat.shape().clone(), new_mat_data);
            let new_bias = mat_mul
                .bias
                .as_ref()
                .map(|old_bias| old_bias.add(&new_bias))
                .unwrap_or(new_bias);

            let weight_matrix = OperandMatrix::new_weight_matrix(new_mat);
            // No config now because we have transposed the matrix, and we no longer need a bias because this would be eliminated when
            // we subtract the mean
            MatMul::new_internal(OperandMatrix::Input, weight_matrix, Some(new_bias), None)
        }
        (OperandMatrix::Input, OperandMatrix::Input) => Err(anyhow::anyhow!(
            "Found MatMul with 2 input matrices, this is not supported"
        )),
    }
}

/// Function to rescale a QKV layer that comes after a LayerNorm
fn rescale_qkv_layer(qkv: &QKV<f32>, scales: &Tensor<f32>, bias: &Tensor<f32>) -> Result<QKV<f32>> {
    let QKV {
        q,
        q_bias,
        k,
        k_bias,
        v,
        v_bias,
        num_heads,
        num_groups,
        ..
    } = qkv;

    // We transform the bias so it is a `1 x bias_size` matrix
    let mut matrix_bias = bias.clone();

    matrix_bias.reshape(Shape::new(vec![1, bias.shape().dim(0)]));

    let mut weights_and_biases = vec![];
    for (old_matrix, old_bias) in [(q, q_bias), (k, k_bias), (v, v_bias)] {
        let new_bias_shape = Shape::new(vec![old_matrix.ncols_2d()]);
        let mut new_bias = matrix_bias.matmul(old_matrix);
        new_bias.reshape(new_bias_shape);

        let new_mat_data = old_matrix
            .slice_last_dim()
            .zip(scales.iter())
            .flat_map(|(row, scale)| row.iter().map(|x| x * scale).collect::<Vec<f32>>())
            .collect::<Vec<f32>>();
        let new_mat = Tensor::new(old_matrix.shape().clone(), new_mat_data);
        // If QKV does not have any bias, then we just take the one given
        let new_bias = old_bias
            .as_ref()
            .map(|bias| bias.add(&new_bias))
            .unwrap_or(new_bias);
        weights_and_biases.push((new_mat, new_bias));
    }
    let (new_v_mat, new_v_bias) = weights_and_biases.pop().unwrap();
    let (new_k_mat, new_k_bias) = weights_and_biases.pop().unwrap();
    let (new_q_mat, new_q_bias) = weights_and_biases.pop().unwrap();

    QKV::new(
        new_q_mat,
        Some(new_q_bias),
        new_k_mat,
        Some(new_k_bias),
        new_v_mat,
        Some(new_v_bias),
        *num_heads,
        *num_groups,
    )
}

#[cfg(test)]
mod tests {
    use ark_std::rand::Rng;
    use tenstore::GenStore;

    use crate::{
        init_test_logging,
        model::{
            ToIterator,
            llm::{Driver, LLMTokenizerObserver},
        },
        parser::{
            file_cache,
            gguf::tests::GPT2_Q8_0,
            llm::{HFTokenizer, LLMTokenizer},
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

            let mul_result = input_matrix.matmul(&modified_const);
            let result_without_mean = input_matrix.matmul(&const_matrix);
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
    fn test_gpt2_replace() -> anyhow::Result<()> {
        init_test_logging("debug");
        // First we load up a GPT-2 model
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let driver = Driver::load_external_model(&model_path)?.with_max_context(10);
        // Extract the model
        let Driver { model, .. } = driver;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);

        let input_tokens = user_tokens
            .into_iter()
            .map(|t| t.as_number::<f32>())
            .collect::<Vec<f32>>();

        let tensor = Tensor::new(vec![input_tokens.len()].into(), input_tokens.clone());
        let shape = tensor.shape();
        let mut store = GenStore::default();

        let trace = model.run::<F>(
            std::slice::from_ref(&tensor),
            Some(vec![shape.clone()]),
            &mut store,
        )?;
        // Get the final node of the Model, we will compare the inputs to this node before and after the transformation (we compare the inputs because the outputs of this layer are tokens
        // and it may be the case that we would get the same tokens out but the actual logits are different)
        let last_model_node_id = model
            .to_backward_iterator()
            .take(1)
            .map(|(id, _)| id)
            .collect::<Vec<NodeId>>()[0];
        // Extract the input to the Logits layer before applying the transformation.
        let pre_transform_final_step = trace.get_step(&last_model_node_id).unwrap();
        let pre_transform_inputs = pre_transform_final_step
            .step_data
            .input_tensors(&mut store)
            .unwrap();
        // Rewrite the model by applying our transformation rule
        let model = LayerNormToRMSNorm.apply(model)?;

        // Now we generate the post-transformation trace and extract the logits step data
        let mut store = GenStore::default();

        let new_trace = model.run::<F>(
            std::slice::from_ref(&tensor),
            Some(vec![shape.clone()]),
            &mut store,
        )?;

        let post_transform_final_step = new_trace.get_step(&last_model_node_id).unwrap();
        let post_transform_inputs = post_transform_final_step
            .step_data
            .input_tensors(&mut store)
            .unwrap();
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
                "Transformed Model output was not close to the original"
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
        let mut driver = Driver::load_external_model(&model_path)?.with_max_context(max_context);
        // Extract the model
        let Driver { model, .. } = driver.clone();
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = HFTokenizer::from_gguf_path(&model_path)?;
        let user_tokens = tokenizer.tokenize(sentence);

        // Rewrite the model by applying our transformation rule
        driver.model = LayerNormToRMSNorm.apply(model)?;

        let driver = driver.into_provable_llm()?;
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            Some(LLMTokenizerObserver {
                input: sentence.to_string(),
                tokenizer: &tokenizer,
            }),
        )?;
        let ctx = driver
            .context::<GoldilocksExt2, Pcs<GoldilocksExt2>>()?
            .with_max_context(max_context);
        let proof = driver.prove(&ctx, trace)?;
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        tracing::info!("Proof size: {}", proof_bytes.len());
        ctx.verify(proof, user_tokens)?;
        Ok(())
    }
}
