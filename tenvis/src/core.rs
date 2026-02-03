use crate::model::{ShapeStep, shape_steps};
use anyhow::Context;
use log::{info, warn};
use std::{collections::HashMap, path::Path, rc::Rc};
use tenstore::GenStore;
use zkml::{
    Element, Number, Shape,
    graph::NodeId,
    inputs::Input,
    model::{
        BaseRunner, HandleLifetimeRunner, Model, StoreRunner, TrackerRunner, llm::Driver,
        tensor_to_handles,
    },
    parser::{
        gguf::{RawGGUF, TensorLoader},
        llm::{HFTokenizer, LLMTokenizer, Token, models::gpt2::GPT2},
        onnx::FloatOnnxLoader,
    },
    quantization::{AbsoluteMax, InferenceTracker, InferenceTrackingMode},
    tensor::TensorTypeParam,
};

#[cfg(test)]
use zkml::model::SanityCheckRunner;

type F = ff_ext::GoldilocksExt2;

pub struct Snapshot<T>
where
    T: TensorTypeParam,
{
    pub model: Model<T>,
    pub store: GenStore,
    pub shapes: HashMap<NodeId, ShapeStep>,
    pub min_max: InferenceTracker,
}

#[allow(dead_code)]
pub struct GlobalContext {
    pub snap_f32: Rc<Snapshot<f32>>,
    pub snap_elt: Rc<Snapshot<Element>>,
}

impl GlobalContext {
    pub fn from_onnx(onnx_file: &Path, input_file: &str) -> anyhow::Result<Self> {
        info!("loading inputs");
        let inputs = Input::from_file(input_file).context("loading input")?;
        if inputs.len() > 1 {
            warn!(
                "{} inputs detected, will only use the first one",
                inputs.len()
            )
        }

        info!("loading model");
        let (model_elt, md) = FloatOnnxLoader::new_with_scaling_strategy(
            onnx_file.as_os_str().to_str().unwrap(),
            AbsoluteMax::new(),
        )
        .with_keep_float(true)
        .build()
        .context("building model from file")?;
        let model_f32 = md.float_model.as_ref().unwrap().clone().to_owned();

        // Run float model
        let snap_f32 = {
            info!("running model in f32 mode");

            let mut store_f32 = GenStore::default();
            let inputs_f32 = model_f32
                .load_input_flat(inputs.as_floats().to_vec())
                .context("preparing inputs for float run")?;
            let input_shapes = inputs_f32
                .iter()
                .map(|x| x.shape().to_owned())
                .collect::<Vec<_>>();
            let input_f32_handles =
                tensor_to_handles(&inputs_f32, model_f32.graph(), &mut store_f32)?;

            let mut min_max_tracker_f32 = InferenceTracker::new(InferenceTrackingMode::MinMax);
            let runner = BaseRunner {
                store: store_f32.clone(),
            };
            let runner = TrackerRunner {
                inner: runner,
                tracker: &mut min_max_tracker_f32,
            };
            #[cfg(test)]
            let runner = SanityCheckRunner { inner: runner };
            let runner = StoreRunner::new(runner, store_f32.clone());
            let mut runner = HandleLifetimeRunner::new(runner, model_f32.graph());

            model_f32
                .run_with_runner(&mut runner, input_f32_handles)
                .context("running the model in float mode")?;

            drop(runner); // ends the TrackerRunner lifetime, so it free the &mut borrow to the tracker.

            Rc::new(Snapshot {
                shapes: shape_steps(model_f32.graph(), &input_shapes)?,
                model: model_f32,
                store: store_f32,
                min_max: min_max_tracker_f32,
            })
        };

        // Run quantized model
        let snap_elt = {
            info!("running model in element mode");

            let mut store_elt = GenStore::default();
            let inputs_elt = model_elt
                .load_input_flat(inputs.clone().to_elements(&md))
                .context("preparing inputs for elt run")?;
            let input_shapes = inputs_elt
                .iter()
                .map(|x| x.shape().to_owned())
                .collect::<Vec<_>>();
            let input_elt_handles =
                tensor_to_handles(&inputs_elt, model_elt.graph(), &mut store_elt)?;

            let mut min_max_tracker_elt = InferenceTracker::new(InferenceTrackingMode::MinMax);
            let runner = BaseRunner {
                store: store_elt.clone(),
            };
            let runner = TrackerRunner {
                inner: runner,
                tracker: &mut min_max_tracker_elt,
            };
            #[cfg(test)]
            let runner = SanityCheckRunner { inner: runner };
            let runner = StoreRunner::new(runner, store_elt.clone());
            let mut runner = HandleLifetimeRunner::new(runner, model_elt.graph());

            model_elt
                .run_with_runner(&mut runner, input_elt_handles)
                .context("running the model in elt mode")?;

            drop(runner); // ends the TrackerRunner lifetime, so it free the &mut borrow to the tracker.

            Rc::new(Snapshot {
                shapes: shape_steps(model_elt.graph(), &input_shapes)?,
                model: model_elt,
                store: store_elt,
                min_max: min_max_tracker_elt,
            })
        };

        Ok(GlobalContext { snap_f32, snap_elt })
    }

    pub fn from_gguf(gguf_file: &Path, prompt: &str, context_size: usize) -> anyhow::Result<Self> {
        info!("loading model");
        let loader = TensorLoader::from_path(gguf_file)?;
        let tokenizer = HFTokenizer::sentencepiece_from_gguf(&loader)?;
        let user_tokens = tokenizer.tokenize(prompt);
        let input_shape = Shape::new(vec![user_tokens.len()]);

        let driver_f32 =
            Driver::load_from_model(GPT2::new(), &RawGGUF::new(gguf_file), Some(context_size))?;

        let snap_f32 = {
            info!("running f32 model");

            let mut min_max_tracker_f32 = InferenceTracker::new(InferenceTrackingMode::MinMax);
            let mut store_f32 = GenStore::default();

            let input_tensor = driver_f32.tokens_to_tensor(&user_tokens)?;
            let input_handles =
                tensor_to_handles(&[input_tensor], driver_f32.model.graph(), &mut store_f32)?;

            let runner = BaseRunner {
                store: store_f32.clone(),
            };
            let runner = TrackerRunner {
                inner: runner,
                tracker: &mut min_max_tracker_f32,
            };
            #[cfg(test)]
            let runner = SanityCheckRunner { inner: runner };
            let runner = StoreRunner::new(runner, store_f32.clone());
            let mut runner = HandleLifetimeRunner::new(runner, driver_f32.model.graph());

            driver_f32.run_with_runner(&mut runner, input_handles)?;

            let outputs = runner.model_outputs(driver_f32.model.graph())?;
            let answer = tokenizer.detokenize(
                outputs
                    .last()
                    .unwrap()
                    .tensor()
                    .unwrap()
                    .data()
                    .iter()
                    .map(|t| Token::from(t.to_usize()))
                    .collect::<Vec<_>>()
                    .as_slice(),
            );

            drop(runner); // ends the TrackerRunner lifetime, so it free the &mut borrow to the tracker.

            info!("f32 result: “{prompt}” -> “{answer}”");
            Rc::new(Snapshot {
                model: driver_f32.model.clone(),
                store: store_f32,
                shapes: shape_steps(driver_f32.model.graph(), std::slice::from_ref(&input_shape))
                    .context("computing shapes for all steps")?,
                min_max: min_max_tracker_f32,
            })
        };

        // NOTE: very important, QKV cache is not reset on clone()
        driver_f32.model.reset();
        let snap_elt = {
            info!("running Element model...");

            let mut min_max_tracker_elt = InferenceTracker::new(InferenceTrackingMode::MinMax);
            let mut store_elt = GenStore::default();
            let (driver_elt, _metadata) = driver_f32.into_provable_llm(None)?;
            let input_tensor = driver_elt.tokens_to_tensor(&user_tokens)?;
            let trace_elt = driver_elt.run_elements_with_tracker::<F>(
                input_tensor,
                &mut store_elt,
                &mut min_max_tracker_elt,
            )?;
            let answer = tokenizer.detokenize(
                trace_elt
                    .outputs()
                    .last()
                    .unwrap()
                    .tensor()
                    .unwrap()
                    .data()
                    .iter()
                    .map(|t| Token::from(t.to_usize()))
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            info!("Element result: “{prompt}” -> “{answer}”");
            Rc::new(Snapshot {
                shapes: shape_steps(driver_elt.model.graph(), &[input_shape])
                    .context("computing shapes for all steps")?,
                model: driver_elt.model,
                store: store_elt,
                min_max: min_max_tracker_elt,
            })
        };

        Ok(GlobalContext { snap_f32, snap_elt })
    }
}
