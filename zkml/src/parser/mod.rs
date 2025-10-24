pub mod gguf;
pub mod json;
pub mod llm;
pub mod onnx;

use crate::{
    Element, Shape,
    layers::{convolution::conv2d_shape, pooling::maxpool2d_shape},
    model::Model,
    padding::pad_model,
    quantization::{AbsoluteMax, ModelMetadata, ScalingStrategy},
};
use anyhow::{Context, Error, Result, bail, ensure};
use itertools::Either;
use tenstore::GenStore;
use tracing::debug;
use tract_onnx::{pb::ModelProto, prelude::*};

/// Utility struct for loading a onnx model with float weights and producing a quantized model
/// that can be used for inference and proving.
#[derive(Debug)]
pub struct FloatOnnxLoader<'a, S: ScalingStrategy> {
    /// Either a path to model file or memmap'd bytes
    model: Either<String, &'a [u8]>,
    scaling_strategy: S,
    model_type: Option<ModelType>,
    keep_float: bool,
}

pub type DefaultFloatOnnxLoader<'a> = FloatOnnxLoader<'a, AbsoluteMax>;

impl DefaultFloatOnnxLoader<'_> {
    pub fn new(model_path: &str) -> Self {
        Self::new_with_scaling_strategy(model_path, AbsoluteMax::new())
    }
}

impl<'a, S: ScalingStrategy> FloatOnnxLoader<'a, S> {
    pub fn new_with_scaling_strategy(model_path: &str, scaling_strategy: S) -> Self {
        Self {
            model: Either::Left(model_path.to_string()),
            scaling_strategy,
            model_type: None,
            keep_float: false,
        }
    }
    pub fn from_bytes_with_scaling_strategy(model_bytes: &'a [u8], scaling_strategy: S) -> Self {
        Self {
            model: Either::Right(model_bytes),
            scaling_strategy,
            model_type: None,
            keep_float: false,
        }
    }

    pub fn with_scaling_strategy(mut self, scaling_strategy: S) -> Self {
        self.scaling_strategy = scaling_strategy;
        self
    }

    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_keep_float(mut self, keep_float: bool) -> Self {
        self.keep_float = keep_float;
        self
    }

    pub fn build(self) -> Result<(Model<Element>, ModelMetadata)> {
        let proto = match self.model {
            Either::Left(path) => load_proto_from_path(&path)?,
            Either::Right(bytes) => {
                use prost_tract_compat::Message;
                ModelProto::decode(bytes)
                    .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))?
            }
        };
        if let Some(model_type) = self.model_type {
            model_type.validate_proto(&proto)?
        }
        let float_model = load_float_model(&proto)?;
        debug!("Input shape: {:?}", float_model.input_shapes());
        let mut kept_float = None;
        if self.keep_float {
            kept_float = Some(float_model.clone());
        }

        // NOTE: this is running with the default store, which is reasonable for the current use.
        // We may wish to change the store type depending on the workload in the future.
        let (quantized_model, mut md) = self
            .scaling_strategy
            .quantize(float_model, &mut GenStore::default())?;
        let padded_model = pad_model(quantized_model)?;
        md.float_model = kept_float;
        Ok((padded_model, md))
    }
}

fn load_proto_from_path(path: &str) -> Result<ModelProto> {
    tract_onnx::onnx()
        .proto_model_for_path(path)
        .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))
}
// Supported operators
const ACTIVATION: [&str; 2] = ["Relu", "Sigmoid"];
const CONVOLUTION: [&str; 1] = ["Conv"];
const DOWNSAMPLING: [&str; 1] = ["MaxPool"];
const LINEAR_ALG: [&str; 2] = ["Gemm", "MatMul"];
const RESHAPE: [&str; 2] = ["Flatten", "Reshape"];

fn is_mlp(model: &ModelProto) -> Result<bool> {
    let mut prev_was_gemm_or_matmul = false;
    let graph = model.graph.as_ref().context("Model has no graph")?;

    for node in graph.node.iter() {
        if LINEAR_ALG.contains(&node.op_type.as_str()) {
            if prev_was_gemm_or_matmul {
                return Ok(false);
            }
            prev_was_gemm_or_matmul = true;
        } else if ACTIVATION.contains(&node.op_type.as_str()) {
            if !prev_was_gemm_or_matmul {
                return Ok(false);
            }
            prev_was_gemm_or_matmul = false;
        } else {
            return Err(Error::msg(format!(
                "Operator '{}' unsupported, yet.",
                node.op_type.as_str()
            )));
        }
    }

    Ok(true)
}

fn is_cnn(model: &ModelProto) -> Result<bool> {
    let mut is_cnn = true;
    let mut found_lin = false;

    let graph = model.graph.as_ref().context("Model has no graph")?;
    let mut previous_op = "";

    for node in graph.node.iter() {
        let op_type = node.op_type.as_str();

        if !CONVOLUTION.contains(&op_type)
            && !DOWNSAMPLING.contains(&op_type)
            && !ACTIVATION.contains(&op_type)
            && !LINEAR_ALG.contains(&op_type)
            && !RESHAPE.contains(&op_type)
        {
            return Err(Error::msg(format!(
                "Operator '{op_type}' unsupported, yet."
            )));
        }

        if ACTIVATION.contains(&op_type) {
            is_cnn =
                is_cnn && (LINEAR_ALG.contains(&previous_op) || CONVOLUTION.contains(&previous_op));
        }

        if DOWNSAMPLING.contains(&op_type) {
            is_cnn = is_cnn && ACTIVATION.contains(&previous_op);
        }

        // Check for dense layers
        if LINEAR_ALG.contains(&op_type) {
            found_lin = true;
        }

        // Conv layers should appear before dense layers
        if found_lin && CONVOLUTION.contains(&op_type) {
            is_cnn = false;
        }
        previous_op = op_type;
        if !is_cnn {
            break;
        }
    }

    Ok(is_cnn)
}

pub fn safe_conv2d_shape(input_shape: &Shape, filter_shape: &Shape) -> Result<Shape> {
    let result = check_filter(filter_shape);
    assert!(result.is_ok(), "conv2d: Failed {:?}", result.unwrap_err());

    check_cnn_input(input_shape).context("conv2d: invalid input shape")?;

    Ok(conv2d_shape(input_shape, filter_shape))
}

pub fn check_filter(filter_shape: &Shape) -> Result<()> {
    ensure!(filter_shape.len() == 4, "Filter should be 4D tensor.");
    ensure!(
        filter_shape[2] == filter_shape[3],
        "Filter should be square."
    );
    Ok(())
}

pub fn check_cnn_input(input_shape: &Shape) -> Result<()> {
    ensure!(input_shape.len() == 3, "input should be 3d tensor");
    ensure!(input_shape[1] == input_shape[2], "input should be square");
    Ok(())
}

pub fn safe_maxpool2d_shape(input_shape: &Shape) -> Result<Shape> {
    check_cnn_input(input_shape).context("maxpool2d: invalid input shape")?;
    Ok(maxpool2d_shape(input_shape))
}

/// Enum representing the different types of models that can be loaded
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    MLP,
    CNN,
}

impl ModelType {
    /// Analyze the given filepath and determine if it matches this model type
    pub fn validate_file(&self, filepath: &str) -> Result<()> {
        let model = load_proto_from_path(filepath)?;
        self.validate_proto(&model)
    }

    /// Analyze the `ModelProto` and determine if it matches this model type
    pub fn validate_proto(&self, model: &ModelProto) -> Result<()> {
        match self {
            ModelType::CNN => {
                if !is_cnn(model)? {
                    bail!("Model is not a valid CNN architecture");
                }
            }
            ModelType::MLP => {
                if !is_mlp(model)? {
                    bail!("Model is not a valid MLP architecture");
                }
            }
        }
        Ok(())
    }

    pub fn from_onnx(filepath: &str) -> Result<ModelType> {
        let model = load_proto_from_path(filepath)?;
        let is_mlp = is_mlp(&model);
        if is_mlp.is_ok() {
            return Ok(ModelType::MLP);
        }
        let is_cnn = is_cnn(&model);
        if is_cnn.is_ok() {
            return Ok(ModelType::CNN);
        }
        bail!(
            "Model is not a valid MLP or CNN architecture: not mlp: {} and not cnn: {}",
            is_mlp.unwrap_err(),
            is_cnn.unwrap_err()
        )
    }
}

/// Unified model loading function that handles both MLP and CNN models
pub fn load_float_model(model: &ModelProto) -> Result<Model<f32>> {
    let model = onnx::from_proto(model)?;
    model.describe();
    Ok(model)
}

// Module for caching downloaded files
pub mod file_cache {
    use anyhow::{Context, ensure};
    use serde::{Deserialize, Serialize};
    use std::{
        io::{BufReader, BufWriter},
        path::PathBuf,
        sync::LazyLock,
    };
    use tracing::warn;

    // Directory to store cached files.
    static CACHE_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
        let dir = PathBuf::from("model_cache");
        if !dir.exists() {
            std::fs::create_dir_all(&dir)
                .expect("Failed to create cache directory for test assets");
        }
        dir
    });

    pub fn from_cache<S: AsRef<str>>(file: S) -> anyhow::Result<PathBuf> {
        let path = CACHE_DIR.join(file.as_ref());
        ensure!(path.exists(), "`{}` not found", path.display());
        ensure!(path.is_file(), "`{}` is not a file", path.display());
        Ok(path)
    }

    /// Attempt to deserialize the `T` contained in `filename`. If `filename`
    /// does not exist in the cache, generate the `T` from `f` and serialize it
    /// into `filename`.
    pub fn deserialize_or_create_with<T: Serialize + for<'a> Deserialize<'a>, S: AsRef<str>>(
        filename: S,
        f: impl Fn() -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        let path = CACHE_DIR.join(filename.as_ref());
        if path.exists() {
            match rmp_serde::from_read::<_, T>(BufReader::new(
                std::fs::File::open(&path)
                    .with_context(|| format!("opening `{}`", path.display()))?,
            ))
            .context("deserializing target value")
            {
                Err(err) => {
                    warn!("failed to retrieve data: {err:?}");
                    warn!("deleting obsolete file `{}` and restarting", path.display());
                    std::fs::remove_file(&path)
                        .with_context(|| format!("deleting `{}`", path.display()))?;
                    deserialize_or_create_with(filename, f)
                }
                t => t,
            }
        } else {
            let t = f()?;
            rmp_serde::encode::write(
                &mut BufWriter::new(
                    std::fs::File::create(&path)
                        .with_context(|| format!("creating `{}`", path.display()))?,
                ),
                &t,
            )
            .context("serializing target value")?;
            Ok(t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Prover, ScalingFactor, init_test_logging_default, quantization::InferenceObserver,
        testing::Pcs, verify,
    };
    use ff_ext::GoldilocksExt2;
    use tenstore::GenStore;
    use tracing::info;
    use transcript::BasicTranscript;

    type F = GoldilocksExt2;

    #[test]
    fn test_load_mlp() {
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let result = FloatOnnxLoader::new(filepath)
            .with_model_type(ModelType::MLP)
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_mlp_model_run() {
        init_test_logging_default();
        let filepath = "assets/scripts/MLP/mlp-iris-01.onnx";
        let (model, md) = FloatOnnxLoader::new(filepath)
            .with_model_type(ModelType::MLP)
            .build()
            .unwrap();
        let input = crate::tensor::Tensor::<f32>::random(&model.input_shapes()[0])
            .to_quantized(md.input_scaling(0));
        let input = model.prepare_inputs(vec![input]).unwrap();
        let trace = model
            .run::<F>(&input, None, &mut GenStore::default())
            .unwrap();
        println!("Result: {:?}", trace.outputs());
    }

    #[test]
    fn test_quantize() {
        let input: [f32; 2] = [0.09039914, -0.07716653];
        let scaling = ScalingFactor::from_span(1.0, -1.0, None);
        println!("Result: {} => {:?}", input[0], scaling.quantize(&input[0]));
        println!("Result: {} => {:?}", input[1], scaling.quantize(&input[0]));
        println!("Result: {} => {:?}", 0, scaling.quantize(&0.0));
        println!("Result: {} => {:?}", -1.0, scaling.quantize(&-1.0));
        println!("Result: {} => {:?}", 1.0, scaling.quantize(&1.0));
    }
    #[test]
    #[ignore]
    fn test_covid_cnn() {
        let subscriber = tracing_subscriber::fmt::Subscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set global subscriber");

        let filepath = "assets/scripts/covid/cnn-covid.onnx";
        let result = FloatOnnxLoader::new(filepath)
            .with_model_type(ModelType::CNN)
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        info!("CREAting random tensor input");
        let (model, md) = result.unwrap();
        let inputs = model
            .unpadded_input_shapes()
            .into_iter()
            .enumerate()
            .map(|(i, shape)| {
                crate::tensor::Tensor::<f32>::random(&shape).to_quantized(md.input_scaling(i))
            })
            .collect();
        let input = model.prepare_inputs(inputs).unwrap();
        info!("RUNNING MODEL...");
        let trace = model
            .run::<F>(&input, None, &mut GenStore::default())
            .unwrap();
        info!("RUNNING MODEL DONE...");
        println!("Result: {:?}", trace.outputs());

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        info!("GENERATING CONTEXT...");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");
        info!("GENERATING CONTEXT DONE...");
        let io = trace.to_verifier_io().unwrap();
        info!("GENERATING Proof...");
        let prover: Prover<'_, '_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&prover_ctx, &mut tr);
        let proof = prover.prove(&trace).expect("unable to generate proof");
        info!("GENERATING Proof DONE...");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");

        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_is_cnn() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let result = is_cnn(&load_proto_from_path(filepath).unwrap());

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());
    }
    #[test]
    fn test_load_cnn() {
        init_test_logging_default();
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        ModelType::CNN.validate_file(filepath).unwrap();
        let result = FloatOnnxLoader::new_with_scaling_strategy(filepath, InferenceObserver::new())
            .with_model_type(ModelType::CNN)
            .build();

        assert!(result.is_ok(), "Failed: {:?}", result.unwrap_err());

        let (model, md) = result.unwrap();
        // let model = pad_model(model).unwrap();
        model.describe();
        let native_input = model
            .unpadded_input_shapes()
            .into_iter()
            .map(|shape| {
                crate::tensor::Tensor::<f32>::random(&shape).to_quantized(md.input_scaling(0))
            })
            .collect();
        let input = model.prepare_inputs(native_input).unwrap();
        let trace = model
            .run::<F>(&input, None, &mut GenStore::default())
            .unwrap();

        let mut tr: BasicTranscript<GoldilocksExt2> = BasicTranscript::new(b"m2vec");
        let (prover_ctx, verifier_ctx) = model
            .generate_contexts::<F, Pcs<F>>()
            .expect("Unable to generate contexts");

        let prover: Prover<'_, '_, GoldilocksExt2, BasicTranscript<GoldilocksExt2>, _> =
            Prover::new(&prover_ctx, &mut tr);
        let io = trace.to_verifier_io().unwrap();
        let proof = prover.prove(&trace).expect("unable to generate proof");
        let mut verifier_transcript: BasicTranscript<GoldilocksExt2> =
            BasicTranscript::new(b"m2vec");
        verify::<_, _, _>(&verifier_ctx, proof, io, &mut verifier_transcript).unwrap();
    }

    #[test]
    fn test_tract() {
        let filepath = "assets/scripts/CNN/cnn-cifar-01.onnx";
        let model = tract_onnx::onnx()
            .model_for_path(filepath)
            .map_err(|e| Error::msg(format!("Failed to load model: {e:?}")))
            .unwrap();
        for symbol in model.symbols.all_symbols().iter() {
            println!("symbol: {symbol:?}");
        }
        let opt = model.into_typed().unwrap();

        let eval_order = opt.eval_order().unwrap();
        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;
            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Cluttered Node: {},  Output {} shape: {:?}",
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
        });

        let opt = opt.into_decluttered().unwrap();

        let eval_order = opt.eval_order().unwrap();

        eval_order.into_iter().for_each(|id| {
            let node = opt.node(id);
            let outputs = &node.outputs;

            for (i, output) in outputs.iter().enumerate() {
                println!(
                    "Node {}: {},  Output {} shape: {:?}",
                    id,
                    node,
                    i,
                    output.fact.shape.dims()
                );
            }
        });

        let mut values = SymbolValues::default();
        let symbol = opt.sym("batch_size");
        values.set(&symbol, 1);

        let opt = opt.concretize_dims(&values).unwrap();
        let plan = SimplePlan::new(opt).unwrap();

        for node_id in plan.order_without_consts() {
            let node = plan.model().node(*node_id);
            println!(
                "planned node {}:{}: input {:?} -> op{:?}",
                node_id,
                node.name,
                node.inputs,
                node.op()
            );
        }
    }
}
