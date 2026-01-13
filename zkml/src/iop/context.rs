use crate::{
    Element, Shape,
    commit::mmcs_context::{CommitmentProverCtx, CommitmentVerifierCtx, GlobalCommitmentContext},
    graph::{Node, NodeId, NodeInput, NodeOutput, order_by_in_port},
    iop::chunking::{ChunkingStrategy, ModelChunk},
    layers::{
        Layer, LayerCtx,
        provable::{OpInfo, ProveInfo},
    },
    lookup::context::{LookupContext, TableType},
    model::{Model, ModelCtx},
    tensor::CommitmentId,
    to_base,
};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::MultilinearExtension, util::ceil_log2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, HashMap};
use tracing::{debug, info_span, trace};
use transcript::Transcript;
use utils::Metrics;

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ProverContext<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub model_ctx: ModelCtx<E>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentProverCtx<E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProverContext<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        self.commitment_ctx.write_to_transcript(t)?;
        Ok(())
    }

    pub fn split_in_chunks<S: ChunkingStrategy>(
        &self,
        num_chunks: Option<usize>,
        strategy: S,
    ) -> anyhow::Result<Vec<ModelChunk>> {
        self.model_ctx.split_in_chunks(num_chunks, &strategy)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct VerifierContext<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub model: ModelCtx<E>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentVerifierCtx<E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifierContext<E, PCS> {
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        self.commitment_ctx.write_to_transcript(t)?;
        Ok(())
    }
}

impl Model<Element> {
    /// Helper method employed to build the context data which is independent from the input shape.
    fn build_global_context_data<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
        input_shapes: &[Shape],
    ) -> anyhow::Result<(ModelCtx<E>, GlobalCommitmentContext<E, PCS>, LookupContext)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        let mut max_poly_len = input_shapes
            .iter()
            .fold(0usize, |acc, shapes| acc.max(shapes.product()));

        // An accumulator used to carry information over while converting the graph
        let mut ctx_aux = ContextAux {
            last_output_shape: input_shapes.to_vec(),
            model_polys: None,
            max_poly_len,
        };

        // TODO: refactor that management of polys
        let mut model_polys = HashMap::<CommitmentId, MultilinearExtension<E>>::new();
        let mut tables = BTreeMap::<TableType, Vec<NodeId>>::new();
        // The shape register is filled along the traversal of the graph.
        let mut shapes: HashMap<NodeOutput, Shape> = HashMap::new();
        debug!("Context : layer info generation ...");
        let graph_ctx = self.graph().try_map_forward(|id, node| {
            Ok(match node {
                Node::Inner(layer) => {
                    let inner_metrics = Metrics::new();
                    // Collect the shapes of the tensor fed to this layer
                    ctx_aux.last_output_shape =
                        order_by_in_port(self.graph().incoming_feeds(id).into_iter().map(|feed| {
                            (
                                NodeInput::new(id, feed.target.port),
                                shapes[&feed.source].clone(),
                            )
                        }))
                        .collect();
                    let layer_ctx =
                        compute_layer_ctx::<E>(&mut ctx_aux, &mut model_polys, id, layer)?;
                    if let Some(lookup_ctx) = layer_ctx.lookup_context() {
                        lookup_ctx
                            .tables
                            .iter()
                            .for_each(|table_type| tables.entry(*table_type).or_default().push(id));
                    }
                    // NOTE: `ctx.last_output_shape` **will** have been modified
                    // by `compute_layer_ctx`, so these shapes are not the one
                    // that have been computed above.
                    shapes.extend(
                        ctx_aux
                            .last_output_shape
                            .iter()
                            .enumerate()
                            .map(|(i, shape)| (NodeOutput::new(id, i), shape.clone())),
                    );

                    max_poly_len = max_poly_len.max(ctx_aux.max_poly_len);
                    debug!(
                        "{} node: {id} ({}), max_poly_len: {max_poly_len}",
                        inner_metrics.to_span(),
                        layer.describe(),
                    );
                    Node::Inner(layer_ctx)
                }
                Node::Input(i) => {
                    // Seed the shape register
                    shapes.insert(
                        NodeOutput::new(self.graph().input_node_id(*i)?, 0),
                        input_shapes[*i].clone(),
                    );
                    Node::Input(*i)
                }
                Node::Output(o) => Node::Output(*o),
            })
        })?;
        // Check to see if we use a lookup table larger than any of the individual polynomials
        tables.keys().for_each(|table_type| {
            let inner_metrics = Metrics::new();
            let multiplicity_vars = table_type.multiplicity_poly_vars();
            max_poly_len = max_poly_len.max(1 << multiplicity_vars);
            debug!(
                "{} table type: {table_type:?}, max_poly_len: {max_poly_len}",
                inner_metrics.to_span()
            );
        });

        let metrics = Metrics::new();
        debug!("Context: lookup generation ...");
        let lookup_ctx = LookupContext::new(&tables);
        debug!("{} lookup generated.", metrics.to_span());

        let metrics = Metrics::new();
        debug!("Context: commitment generating ...");
        let commitment_ctx = GlobalCommitmentContext::<E, PCS>::new(
            max_poly_len,
            model_polys,
            &lookup_ctx.tables.keys().collect_vec(),
            graph_ctx.next_node_id(),
        )?;
        debug!("{} commitment generated.", metrics.to_span());
        Ok((ModelCtx::new(graph_ctx), commitment_ctx, lookup_ctx))
    }

    /// Generate the prover and verifier contexts for the input shape embedded in the model
    pub fn generate_contexts<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
    ) -> anyhow::Result<(ProverContext<E, PCS>, VerifierContext<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        let input_shapes = self.input_shapes();
        let span = info_span!("zkml_generate_contexts", inputs = input_shapes.len());
        let _guard = span.enter();
        let (step_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        let (prover_ctx, verifier_ctx) = commitment_ctx.generate_contexts()?;

        let prover_ctx = ProverContext {
            model_ctx: step_info.clone(),
            commitment_ctx: prover_ctx,
            lookup: lookup.clone(),
        };

        let verifier_ctx = VerifierContext {
            model: step_info,
            commitment_ctx: verifier_ctx,
            lookup,
        };

        Ok((prover_ctx, verifier_ctx))
    }

    /// Generate the prover and verifier contexts for shapes provided as input.
    /// For models with variable input shapes (like LLMs) the maximum possible input shape should be passed here.
    /// Given the maximum possible input shape this method constructs prover and verifier contexts suitable for all
    /// acceptable inputs.
    pub fn generate_contexts_for_input_shapes<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        input_shapes: Vec<Shape>,
    ) -> anyhow::Result<(ProverContext<E, PCS>, VerifierContext<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        let span = info_span!(
            "zkml_generate_contexts_for_input_shapes",
            inputs = input_shapes.len()
        );
        let _guard = span.enter();
        debug!("Building global context");
        let (steps_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        debug!("Built global context");
        debug!("Building all contexts");
        let (commit_prover_ctx, commit_verifier_ctx) = commitment_ctx.generate_contexts()?;
        let prover_context = ProverContext {
            model_ctx: steps_info.clone(),
            commitment_ctx: commit_prover_ctx,
            lookup: lookup.clone(),
        };

        let verifier_context = VerifierContext {
            model: steps_info,
            commitment_ctx: commit_verifier_ctx,
            lookup,
        };
        Ok((prover_context, verifier_context))
    }
}
/// Similar to the InferenceStep but only records the input and output shapes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeStep {
    pub unpadded_input_shape: Vec<Shape>,
    pub unpadded_output_shape: Vec<Shape>,
    pub padded_input_shape: Vec<Shape>,
    pub padded_output_shape: Vec<Shape>,
}

impl ShapeStep {
    pub fn new(
        unpadded_input: Vec<Shape>,
        padded_input: Vec<Shape>,
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
    ) -> ShapeStep {
        Self {
            unpadded_input_shape: unpadded_input,
            padded_input_shape: padded_input,
            unpadded_output_shape: unpadded_output,
            padded_output_shape: padded_output,
        }
    }
    pub fn next_step(
        last_step: &ShapeStep,
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
    ) -> ShapeStep {
        ShapeStep {
            unpadded_input_shape: last_step.unpadded_output_shape.clone(),
            unpadded_output_shape: unpadded_output,
            padded_input_shape: last_step.padded_output_shape.clone(),
            padded_output_shape: padded_output,
        }
    }
}

/// Auxiliary information for the context creation
#[derive(Clone, Debug)]
pub struct ContextAux {
    pub last_output_shape: Vec<Shape>,
    pub model_polys: Option<HashMap<CommitmentId, Vec<Element>>>,
    /// This field is only used in macro layers like MHA
    pub max_poly_len: usize,
}

fn compute_layer_ctx<E: ExtensionField>(
    ctx_aux: &mut ContextAux,
    model_polys: &mut HashMap<CommitmentId, MultilinearExtension<E>>,
    id: NodeId,
    layer: &Layer<Element>,
) -> anyhow::Result<LayerCtx<E>> {
    trace!(
        "Context : {}-th layer {}info generation ...",
        id,
        layer.describe()
    );
    let (info, mut new_aux) = layer.step_info(id, ctx_aux.clone())?;
    // Retrieve any model polynomials that need to be committed
    if new_aux.model_polys.is_some() {
        for (poly_id, evals) in new_aux.model_polys.as_mut().unwrap().drain() {
            let num_vars = ceil_log2(evals.len());
            let mle =
                MultilinearExtension::<E>::from_evaluations_vec(num_vars, to_base::<E, _>(evals));
            if let Some(expected_mle) = model_polys.get(&poly_id) {
                // check that the same poly was stored for `poly_id`
                debug_assert!(
                    expected_mle == &mle,
                    "Found different MLE for polynomial {poly_id}"
                );
            } else {
                model_polys.insert(poly_id, mle);
            }
        }
    }
    *ctx_aux = new_aux;
    Ok(info)
}
