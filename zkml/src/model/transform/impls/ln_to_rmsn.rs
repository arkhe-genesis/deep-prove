//! Definition of the [`RewriteRule`] to replace [`LayerNorm`] with [`RMSNorm`]
use crate::{
    Tensor,
    graph::{Direction, NodeId},
    layers::{
        Layer,
        add::ADD_LAYER,
        matrix_mul::{Config, MATMUL_LAYER, MatMul, OperandMatrix},
        transformer::{
            embeddings::Embeddings,
            layernorm::LayerNorm,
            positional::{POSITIONAL_LAYER, Positional, PositionalVariant, absolute::Absolute},
            qkv::QKV,
            rmsnorm::RMSNorm,
        },
    },
    model::{Model, transform::ModelTransform},
    shape::Shape,
    tensor::KeyedTensor,
};
use anyhow::{Result, anyhow, bail, ensure};

#[derive(Debug)]
/// Rewrite rule to replace LayerNorm with RMSNorm, currently this transformation should only be used with GPT2 Models.
pub struct LayerNormToRMSNorm;

impl ModelTransform<f32> for LayerNormToRMSNorm {
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
/// either Add, [`Positional`] or [`MatMul`] and that there is at least one [`MatMul`].
fn add_was_previous_layer(model: &mut Model<f32>, input_node_id: NodeId) -> Result<()> {
    let input_ids = model
        .graph
        .incomings(input_node_id)
        .map(|(_, edge)| edge.source())
        .filter(|n_id| model.graph[*n_id].as_inner().is_some())
        .collect::<Vec<_>>();

    let seen_a_matmul = input_ids
        .into_iter()
        .try_fold(false, |seen_a_matmul, input_id| {
            // safe unwrap since it's guaranteed to be a node because we used node_neighbors
            let Some(input_node) = model.graph.node_mut(input_id).unwrap().as_inner_mut() else {
                unreachable!("filtered above")
            };
            let add_input_op_name = input_node.short_name();
            match add_input_op_name {
                MATMUL_LAYER => {
                    // Found a LayerNorm node with an Add layer as input, which has a linear layer as input
                    modify_matrix_subtract_mean(input_node)?;
                    Ok(true)
                }
                ADD_LAYER | POSITIONAL_LAYER => Ok(seen_a_matmul),
                _ => bail!("Expected MatMul or Add layer, found {add_input_op_name}"),
            }
        })?;
    ensure!(
        seen_a_matmul,
        "Expected to find a MatMul layer as input to the Add {} layer before LayerNorm, found none",
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
        Layer::<f32>::MatMul(mat_mul) => modify_matmul(mat_mul),
        Layer::<f32>::Positional(positional) => modify_positional(positional),
        Layer::<f32>::Embeddings(embeddings) => modify_embeddings(embeddings),
        other => bail!(
            "Expected MatMul, Positional or Embeddings operation, found {}",
            other.short_name()
        ),
    }
}

/// Modify the constant matrix in a [`MatMul`] layer so that the output has rows with mean 0.
fn modify_matmul(mat_mul: &mut MatMul<f32>) -> Result<()> {
    match (&mat_mul.left_matrix, &mat_mul.right_matrix) {
        (OperandMatrix::Weight(_), OperandMatrix::Weight(_)) => Err(anyhow!(
            "Found layer with 2 constant matrices, which is useless as the
                product can be directly used instead"
        )),
        (OperandMatrix::Weight(..), OperandMatrix::Input) => Err(anyhow!(
            "Found MatMul with constant left matrix, this is not supported"
        )),
        (OperandMatrix::Input, OperandMatrix::Weight(mat)) => {
            let new_mat = mat.tensor.try_new_map_tensor(|t| {
                if let Some(Config::TransposeB) = &mat_mul.config {
                    Ok(mean_subtracted_matrix(&t.transpose()?)?)
                } else {
                    Ok(mean_subtracted_matrix(t)?)
                }
            })?;

            let weight_matrix = OperandMatrix::new_weight_matrix(new_mat);
            // Now we subtract the bias mean from each element of the bias
            let new_bias = mat_mul
                .bias
                .as_ref()
                .map(|old_bias| {
                    old_bias.try_new_map_tensor(|bias_tensor| {
                        let bias_shape = bias_tensor.shape();
                        let bias_sum = bias_tensor.iter().sum::<f32>();
                        let bias_mean = bias_sum / bias_shape.dim(0) as f32;
                        let new_bias_data = bias_tensor
                            .iter()
                            .map(|x| x - bias_mean)
                            .collect::<Vec<f32>>();
                        Tensor::new(bias_shape.clone(), new_bias_data)
                    })
                })
                .transpose()?;

            // No config now because we have transposed the matrix,
            *mat_mul = MatMul::new_internal(OperandMatrix::Input, weight_matrix, new_bias, None)?;
            Ok(())
        }
        (OperandMatrix::Input, OperandMatrix::Input) => Err(anyhow::anyhow!(
            "Found MatMul with 2 input matrices, this is not supported"
        )),
    }
}
/// Modify the embeddings in an [`Embeddings`] layer so that the output has rows with mean 0.
fn modify_embeddings(embeddings: &mut Embeddings<f32>) -> Result<()> {
    // The embedding is just a wrapper around a MatMul with extra info so we call modify_matmul
    modify_matmul(&mut embeddings.mat)
}

/// Modify the positional encodings in a [`Positional`] layer so that the output has rows with mean 0.
fn modify_positional(positional_layer: &mut Positional<f32>) -> Result<()> {
    // Match on the type of positional encoding, we expect `Learned` here
    *positional_layer = match &positional_layer.variant {
        PositionalVariant::Absolute(absolute) => {
            let Absolute::<f32> { positional, .. } = absolute;
            let new_mat = positional.try_new_map_tensor(mean_subtracted_matrix)?;
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
fn mean_subtracted_matrix(matrix: &Tensor<f32>) -> Result<Tensor<f32>> {
    let matrix_shape = matrix.shape();
    let row_size = matrix_shape.dim(matrix_shape.rank() - 1);

    let subtract_mean_matrix = (0..row_size)
        .flat_map(|i| (0..row_size).map(move |j| if i == j { row_size as f32 - 1.0 } else { -1.0 }))
        .collect::<Vec<f32>>();

    let subtract_mean_tensor =
        Tensor::new(Shape::new(vec![row_size, row_size]), subtract_mean_matrix)?;
    let mut modified_matrix = matrix.matmul(&subtract_mean_tensor)?;
    modified_matrix
        .iter_mut()
        .for_each(|x| *x /= row_size as f32);
    Ok(modified_matrix)
}

/// This function is used to modify a [`MatMul`] or [`QKV`] layer to absorb the weights and biases from the preceding [`LayerNorm`].
fn modify_subsequent_linear_layer(
    node: &mut Layer<f32>,
    weights: &Tensor<f32>,
    bias: &KeyedTensor<f32>,
) -> Result<()> {
    *node = match &node {
        Layer::<f32>::MatMul(mat_mul) => Layer::MatMul(rescale_matmul(mat_mul, weights, bias)?),
        Layer::<f32>::QKV(qkv) => Layer::QKV(rescale_qkv_layer(qkv, weights, bias)?),
        other => bail!(
            "Expected MatMul or QKV operation, found {}",
            other.short_name()
        ),
    };
    Ok(())
}

/// Function that rescales the weight matrix and modifies the bias of a [`MatMul`] layer.
fn rescale_matmul(
    mat_mul: &MatMul<f32>,
    scales: &Tensor<f32>,
    bias: &KeyedTensor<f32>,
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
                mat.tensor.transpose()?
            } else {
                mat.tensor.tensor()
            };
            // We transform the bias so it is a `1 x bias_size` matrix
            let new_bias = bias.try_new_map_tensor(|bias| {
                let mut matrix_bias = bias.clone();
                matrix_bias.reshape(Shape::new(vec![1, bias.shape().dim(0)]))?;
                let new_bias_shape = Shape::new(vec![inner_mat.ncols_2d()?]);
                let mut new_bias = matrix_bias.matmul(&inner_mat)?;
                new_bias.reshape(new_bias_shape)?;
                Ok(new_bias)
            })?;

            let new_mat_data = inner_mat
                .slice_last_dim()
                .zip(scales.iter())
                .flat_map(|(row, scale)| row.iter().map(|x| x * scale).collect::<Vec<f32>>())
                .collect::<Vec<f32>>();
            let new_mat = KeyedTensor::new(
                mat.tensor.key.clone(),
                Tensor::new(inner_mat.shape().clone(), new_mat_data)?,
            );
            let new_bias = mat_mul
                .bias
                .as_ref()
                .map(|old_bias| old_bias.new_map_tensor(|bias| bias.add(&new_bias)))
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
fn rescale_qkv_layer(
    qkv: &QKV<f32>,
    scales: &Tensor<f32>,
    bias: &KeyedTensor<f32>,
) -> Result<QKV<f32>> {
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
    let mut matrix_bias = bias.tensor();

    matrix_bias.reshape(Shape::new(vec![1, bias.shape().dim(0)]))?;

    let mut weights_and_biases = vec![];
    for (old_matrix, old_bias) in [(q, q_bias), (k, k_bias), (v, v_bias)] {
        let matrix_tensor = old_matrix;
        let new_bias_shape = Shape::new(vec![matrix_tensor.ncols_2d()?]);
        let mut new_bias = matrix_bias.matmul(matrix_tensor)?;
        new_bias.reshape(new_bias_shape)?;
        let new_bias = KeyedTensor::new(bias.key.clone(), new_bias);

        let new_mat_data = matrix_tensor
            .slice_last_dim()
            .zip(scales.iter())
            .flat_map(|(row, scale)| row.iter().map(|x| x * scale).collect::<Vec<f32>>())
            .collect::<Vec<f32>>();
        let new_mat = KeyedTensor::new(
            old_matrix.key.clone(),
            Tensor::new(old_matrix.shape().clone(), new_mat_data)?,
        );
        // If QKV does not have any bias, then we just take the one given
        let new_bias = old_bias
            .as_ref()
            .map(|bias| bias.new_map_tensor(|bias| bias.add(&new_bias)))
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
        model::llm::{Driver, LLMTokenizerObserver},
        parser::{
            default_pipeline_config, file_cache,
            gguf::RawGGUF,
            llm::{
                LLMTokenizer,
                models::gpt2::{GPT2, tests::GPT2_Q8_0},
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
        let modified_const = mean_subtracted_matrix(&const_matrix).unwrap();

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
        let model = LayerNormToRMSNorm.apply(model)?;

        // Now we generate the post-transformation trace and extract the logits step data
        let mut store = GenStore::default();

        let new_trace = model.run::<F>(vec![tensor], &mut store)?;

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
        let gguf = RawGGUF::new(model_path);
        let driver = Driver::load_from_model(GPT2, &gguf, Some(max_context))?;
        // Make a tester input for the model so we can compare the pre and post transformation outputs
        let sentence = "The sky is";
        let tokenizer = GPT2.load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);

        // Rewrite the model by applying our transformation rule
        let conf = default_pipeline_config().with_float_rules(vec![Box::new(LayerNormToRMSNorm)]);

        let driver = driver.into_provable_llm(Some(conf))?;
        let trace = driver.run::<GoldilocksExt2>(
            user_tokens.clone(),
            &mut GenStore::default(),
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
