pub mod gguf;
pub mod json;
pub mod llm;
pub mod onnx;
pub mod safe;

use crate::{
    Element, ScalingStrategy, Shape, model::Model, padding::pad_model,
    quantization::InferenceObserver,
};

use crate::model::transform::ModelTransform;
use tenstore::GenStore;

/// A trait for data formats that can provide metadata about the model.
pub trait ModelNameProvider {
    /// It can return a vec since there are multiple information related to the model.
    fn model_metadata(&self) -> anyhow::Result<Vec<String>>;
}
/// Loading a model from raw data requires passing the data through a whole pipeline
/// with several steps. This struct is used to configure the different steps of the pipeline.
#[derive(Default)]
pub struct PipelineConfig<'a, S> {
    float_rules: Vec<Box<dyn ModelTransform<f32>>>,
    quantized_rules: Vec<Box<dyn ModelTransform<Element>>>,
    input_shapes: Option<Vec<Shape>>,
    quant_strategy: Option<S>,
    store: Option<&'a mut GenStore>,
}

pub fn default_pipeline_config() -> PipelineConfig<'static, InferenceObserver> {
    PipelineConfig::default()
}

impl<'a, S: ScalingStrategy> PipelineConfig<'a, S> {
    pub fn with_float_rules(mut self, rules: Vec<Box<dyn ModelTransform<f32>>>) -> Self {
        self.float_rules = rules;
        self
    }
    pub fn with_quantized_rules(mut self, rules: Vec<Box<dyn ModelTransform<Element>>>) -> Self {
        self.quantized_rules = rules;
        self
    }
    pub fn with_input_shapes(mut self, input_shapes: Vec<Shape>) -> Self {
        self.input_shapes = Some(input_shapes);
        self
    }
    pub fn with_strategy(mut self, strategy: S) -> Self {
        self.quant_strategy = Some(strategy);
        self
    }
    pub fn with_store(mut self, store: &'a mut GenStore) -> Self {
        self.store = Some(store);
        self
    }
}
///// Trait for loading a model from a given format.
pub trait ModelLoader<DataFormat> {
    /// The configuration of the model.
    /// Often, there are some subtle parameters that are not reflected directly in the "graph" of the model itself.
    /// For example, for LLM models, the configuration is the LLMConfig that contains information like the maximum
    /// context length, the vocabulary size etc. For generic ONNX models, there is no external configuration.
    type ModelConfig;
    /// The main method a loader must implement. A model loader can only parse for the explicit format of the model it supports.
    /// For example, Gemma3 derived models can be loaded from `[GGUFFormat]` format or `[SafeTensorsFormat]` format.
    /// Note the returned model is only loaded in float. One must use `[to_quantized]` to quantize the model into a model
    /// ready to proven.
    fn parse(&self, raw: &DataFormat) -> anyhow::Result<(Model<f32>, Self::ModelConfig)>;
    fn model_name(&self) -> String;
}

/// Convert a float model into a quantized model using the given pipeline configuration.
pub fn to_quantized<S: ScalingStrategy>(
    mut model: Model<f32>,
    mut pipeline_config: PipelineConfig<S>,
) -> anyhow::Result<Model<Element>> {
    // 1. set the input shapes
    if let Some(input_shapes) = pipeline_config.input_shapes.take() {
        model.input_shapes = input_shapes.clone();
        // NOTE: currently no difference between padded and unpadded input shapes as it's
        // mostly used for LLM and this notion of padded/unpadded should disappear soon
        model.unpadded_input_shapes = input_shapes.clone();
    }
    let mut default_store = GenStore::default();
    let default_strategy = InferenceObserver::new();
    let store = if let Some(store) = pipeline_config.store {
        store
    } else {
        &mut default_store
    };
    // 2. apply float rules
    for rule in pipeline_config.float_rules {
        model = rule.apply(model)?;
    }
    // 3. quantize the model into Elements
    // NOTE: we could return but is it useful ?
    let (mut quantized_model, _md) = if let Some(strategy) = pipeline_config.quant_strategy {
        strategy.quantize(model, store)?
    } else {
        default_strategy.quantize(model, store)?
    };
    // 4. apply quantized rules
    for rule in pipeline_config.quantized_rules {
        quantized_model = rule.apply(quantized_model)?;
    }
    // 5. pad the model
    let padded_model = pad_model(quantized_model)?;
    Ok(padded_model)
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

    pub fn cache_path<S: AsRef<str>>(file: S) -> PathBuf {
        CACHE_DIR.join(file.as_ref())
    }

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
