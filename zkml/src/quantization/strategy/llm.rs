//! LLM quantisation strategy for Deep Prove.

use super::*;
use crate::{
    layers::requant::FIXED_POINT_SCALE,
    lookup::table::SHIFT_CHECK_TABLE_BIT_SIZE,
    parser::llm::{
        HFTokenizer, LLMTokenizer, Token, metadata::LLMMetadata, tokenizer::TokenizerLoader,
    },
    quantization::strategy::llm_tracker::{LLMInferenceTracker, LLMTrackerRunner},
};

use dp_crypto::util::ceil_log2;
use serde::{Deserialize, Serialize};

pub const fn get_default_prompt() -> &'static str {
    "Artificial intelligence and machine learning have revolutionized numerous industries by enabling computers to perform 
    tasks that traditionally required human intelligence, such as natural language processing, computer vision, predictive analytics, 
    and autonomous decision-making, while simultaneously creating new opportunities for innovation in healthcare through diagnostic imaging 
    and drug discovery, in finance through algorithmic trading and fraud detection, in transportation through self-driving vehicles and route 
    optimization, in entertainment through recommendation systems and content generation, in education through personalized learning platforms 
    and automated grading systems, in manufacturing through predictive maintenance and quality control, in agriculture through precision farming 
    and crop monitoring, in cybersecurity through threat detection and response automation, in retail through inventory management and customer 
    behavior analysis, and in scientific research through data analysis and hypothesis generation, thereby demonstrating the transformative potential 
    of these technologies to augment human capabilities, improve efficiency, reduce costs, enhance accuracy, and solve complex problems across 
    diverse domains while also raising important questions about ethical considerations, privacy protection, job displacement, algorithmic bias, 
    transparency, accountability, and the need for responsible development and deployment of AI systems that benefit society as a whole rather 
    than exacerbating existing inequalities or creating new risks."
}
/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMInferenceObserver {
    quantising_prompt: Option<String>,
    tokeniser: HFTokenizer,
    llm_metadata: LLMMetadata,
}

impl LLMInferenceObserver {
    pub fn new<Model, R>(
        model: Model,
        raw: &R,
        llm_metadata: LLMMetadata,
        prompt: Option<&str>,
    ) -> Self
    where
        Model: TokenizerLoader<R>,
    {
        Self {
            quantising_prompt: prompt.map(|s| s.to_owned()),
            tokeniser: model.load_tokenizer(raw).expect("Failed to load tokenizer"),
            llm_metadata,
        }
    }

    pub fn new_with_tokeniser(
        tokeniser: HFTokenizer,
        llm_metadata: LLMMetadata,
        prompt: Option<&str>,
    ) -> Self {
        Self {
            quantising_prompt: prompt.map(|s| s.to_owned()),
            tokeniser,
            llm_metadata,
        }
    }

    pub fn make_representative_input(&self, max_context_length: usize) -> Vec<Token> {
        let prompt = self
            .quantising_prompt
            .as_deref()
            .unwrap_or(get_default_prompt());
        let mut input_ids = self.tokeniser.tokenize(prompt);
        if input_ids.len() >= max_context_length {
            input_ids.truncate(max_context_length);
        } else {
            input_ids = input_ids
                .into_iter()
                .cycle()
                .take(max_context_length)
                .collect::<Vec<Token>>();
        }
        input_ids
    }

    pub fn get_representative_input_length(&self) -> usize {
        let prompt = self
            .quantising_prompt
            .as_deref()
            .unwrap_or(get_default_prompt());
        let token_ids = self.tokeniser.tokenize(prompt);
        token_ids.len()
    }
}

impl ScalingStrategy for LLMInferenceObserver {
    type AuxData = LLMInferenceTracker;

    fn quantize(
        &self,
        model: Model<f32>,
        store: &mut GenStore,
    ) -> Result<(Model<Element>, ModelMetadata)> {
        let mut tracker = LLMInferenceTracker::new(self.llm_metadata.clone());

        let unpadded_input_shapes = model.input_shapes();
        let context_length = unpadded_input_shapes[0].numel();
        let representative_input_tokens = self.make_representative_input(context_length);
        let input_tensor = Tensor::<f32>::new(
            unpadded_input_shapes[0].clone(),
            representative_input_tokens
                .iter()
                .map(|t| t.as_tensor_type_param::<f32>())
                .collect::<Vec<f32>>(),
        )?;

        let input_node_id = model.graph().input_node_id(0)?;

        let handle = TensorHandle::from_tensor(
            input_node_id.output_at(0).to_storage_key(),
            store.clone(),
            input_tensor,
        );

        let runner = BaseRunner::from(store.clone());
        let runner = LLMTrackerRunner {
            inner: runner,
            tracker: &mut tracker,
        };
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };
        let mut runner = HandleLifetimeRunner::new(runner, model.graph());
        model.run_with_runner(&mut runner, vec![handle])?;

        // Get the scaling factor of the input
        let num_model_inputs = unpadded_input_shapes.len();
        let input_scaling = (0..num_model_inputs)
            .map(|i| {
                let input_node_id = model.graph().input_node_id(i)?;
                let (input_min, input_max) =
                    tracker.scaling_range(input_node_id.output_at(0)).unwrap();
                Ok(ScalingFactor::from_absolute_max(
                    input_min.abs().max(input_max.abs()),
                    None,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        quantize_model::<LLMInferenceObserver>(model, tracker, input_scaling)
    }

    fn scaling_factors_for_node(
        tracker: &LLMInferenceTracker,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        let llm_metadata = tracker.llm_metadata();
        if let Some(positional_id) = llm_metadata.positional
            && positional_id == node_id
        {
            // In this case we want to add every row of the positional embedding matrix to the embedding matrix and find the max value
            let embeddings_max = tracker.calculate_embeddings_max().unwrap();
            let shift = 23;
            let domain_min = -1 << shift;
            let domain_max = (1 << shift) - 1;
            let sf =
                ScalingFactor::from_absolute_max(embeddings_max, Some((domain_min, domain_max)));
            return vec![sf];
        }

        let is_activation = llm_metadata
            .transformers
            .iter()
            .any(|block| block.ffn.activation_id == node_id);
        (0..num_outputs)
            .map(|i| {
                let node_output = node_id.output_at(i);
                let (min, max) = tracker.scaling_range(node_output).unwrap();
                if !tracker.is_add_layer(node_output) {
                    ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
                } else if is_activation {
                    let contraction_size = ceil_log2(llm_metadata.config.intermediate_size);
                    let max_bits = 63usize;
                    let available_bits =
                        max_bits - contraction_size - *quantization::BIT_LEN - FIXED_POINT_SCALE;
                    let bit_size = (available_bits - 1).min(SHIFT_CHECK_TABLE_BIT_SIZE);
                    ScalingFactor::from_absolute_max(
                        min.abs().max(max.abs()),
                        Some((-1 << bit_size, (1 << bit_size) - 1)),
                    )
                } else {
                    // In this case we are in an Add layer, so we know its the residual that we wish to keep in higher precision.
                    // We pick the domain size to be the largest multiple of quantization::BIT_LEN that is less than or equal to 24.
                    let mut multiple = 1usize;
                    while (multiple + 1) * *quantization::BIT_LEN <= 24 {
                        multiple += 1;
                    }
                    let multiple_bit_size = multiple * *quantization::BIT_LEN;
                    let domain_min = -1 << (multiple_bit_size - 1);
                    let domain_max = (1 << (multiple_bit_size - 1)) - 1;
                    ScalingFactor::from_absolute_max(
                        min.abs().max(max.abs()),
                        Some((domain_min, domain_max)),
                    )
                }
            })
            .collect()
    }

    fn scaling_factor_for_intermediate_data(
        tracker: &LLMInferenceTracker,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        tracker.scaling_factor_for_intermediate_data(node_id, data_id)
    }

    fn name(&self) -> String {
        "LLMInferenceObserver".to_string()
    }
}
