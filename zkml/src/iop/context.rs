use crate::{
    Element, Shape,
    commit::mmcs_context::{
        CommitmentProverCtx, CommitmentVerifierCtx, GlobalCommitmentContext, PolyId,
    },
    layers::provable::{Node, NodeCtx, NodeId, OpInfo},
    lookup::context::{LookupContext, TableType},
    model::{Model, ModelCtx, ToIterator},
    to_base,
};
use anyhow::{Ok, anyhow, ensure};
use ff_ext::ExtensionField;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::{mle::MultilinearExtension, util::ceil_log2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use tracing::{debug, trace};
use transcript::Transcript;
use utils::Metrics;

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ProverContext<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx<E>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentProverCtx<E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
    /// unpadded shape of the first initial input
    pub unpadded_input_shapes: Vec<Shape>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProverContext<E, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
{
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        self.commitment_ctx.write_to_transcript(t)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct VerifierContext<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx<E>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentVerifierCtx<E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
    /// unpadded shape of the first initial input
    pub unpadded_input_shapes: Vec<Shape>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> VerifierContext<E, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
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
        let tables = BTreeSet::new();
        let mut max_poly_len = input_shapes
            .iter()
            .fold(0usize, |acc, shapes| acc.max(shapes.product()));

        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape: input_shapes.to_vec(),
            model_polys: None,
            max_poly_len,
        };

        let (step_infos, commitment_ctx, lookup) = {
            let mut model_polys = Vec::<(NodeId, HashMap<PolyId, MultilinearExtension<E>>)>::new();
            let mut step_infos = BTreeMap::new();
            let mut shapes: HashMap<NodeId, Vec<Shape>> = HashMap::new();
            debug!("Context : layer info generation ...");
            let mut max_node_id = NodeId(0);
            for (id, node) in self.to_forward_iterator() {
                let inner_metrics = Metrics::new();
                ctx_aux = compute_node_shape::<E>(
                    ctx_aux,
                    &mut model_polys,
                    &mut step_infos,
                    &mut shapes,
                    input_shapes,
                    id,
                    node,
                )?;
                max_poly_len = max_poly_len.max(ctx_aux.max_poly_len);
                max_node_id = max_node_id.max(id);
                debug!(
                    "{} node: {id} ({}), max_poly_len: {max_poly_len}",
                    inner_metrics.to_span(),
                    node.describe(),
                );
            }
            // Check to see if we use a lookup table alrger than any of the individual polynomials
            ctx_aux.tables.iter().for_each(|table_type| {
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
            let lookup_ctx = LookupContext::new(&ctx_aux.tables);
            debug!("{} lookup generated.", metrics.to_span());

            let metrics = Metrics::new();
            debug!("Context: commitment generating ...");
            let commitment_ctx = GlobalCommitmentContext::<E, PCS>::new(
                max_poly_len,
                model_polys,
                &lookup_ctx.tables,
                max_node_id,
            )?;
            debug!("{} commitment generated.", metrics.to_span());
            (step_infos, commitment_ctx, lookup_ctx)
        };

        Ok((ModelCtx { nodes: step_infos }, commitment_ctx, lookup))
    }

    /// Compute the size of the biggest polynomial to be committed for the given `input_shape`
    pub(crate) fn compute_max_poly_size<E: ExtensionField>(
        &self,
        input_shapes: &[Shape],
    ) -> anyhow::Result<usize> {
        let mut max_poly_len = input_shapes
            .iter()
            .fold(0usize, |acc, shapes| acc.max(shapes.product()));

        let mut ctx_aux = ContextAux {
            tables: BTreeSet::new(),
            last_output_shape: input_shapes.to_vec(),
            model_polys: None,
            max_poly_len,
        };
        let mut shapes = HashMap::new();
        for (id, node) in self.to_forward_iterator() {
            let node_input_shapes = compute_node_input_shapes(input_shapes, &shapes, id, node)?;
            ctx_aux.last_output_shape = node_input_shapes;
            let (_, new_aux) = node.step_info::<E>(id, ctx_aux)?;
            shapes.insert(id, new_aux.last_output_shape.clone());
            max_poly_len = max_poly_len.max(new_aux.max_poly_len);
            ctx_aux = new_aux;
        }
        Ok(max_poly_len)
    }

    /// Generate the prover and verifier contexts for the input shape embedded in the model
    pub fn generate_contexts<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
    ) -> anyhow::Result<(ProverContext<E, PCS>, VerifierContext<E, PCS>)>
    where
        PCS::CommitmentWithWitness: Serialize + DeserializeOwned,
    {
        let input_shapes = self.input_shapes();
        let (step_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        let (prover_ctx, verifier_ctx) = commitment_ctx.generate_contexts()?;

        let prover_ctx = ProverContext {
            steps_info: step_info.clone(),
            commitment_ctx: prover_ctx,
            lookup: lookup.clone(),
            unpadded_input_shapes: self.unpadded_input_shapes(),
        };

        let verifier_ctx = VerifierContext {
            steps_info: step_info,
            commitment_ctx: verifier_ctx,
            lookup,
            unpadded_input_shapes: self.unpadded_input_shapes(),
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
        debug!("Building global context");
        let (steps_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        debug!("Built global context");
        debug!("Building all contexts");
        let (commit_prover_ctx, commit_verifier_ctx) = commitment_ctx.generate_contexts()?;
        let prover_context = ProverContext {
            steps_info: steps_info.clone(),
            commitment_ctx: commit_prover_ctx,
            lookup: lookup.clone(),
            unpadded_input_shapes: self.unpadded_input_shapes(),
        };

        let verifier_context = VerifierContext {
            steps_info,
            commitment_ctx: commit_verifier_ctx,
            lookup,
            unpadded_input_shapes: self.unpadded_input_shapes(),
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
    pub tables: BTreeSet<TableType>,
    pub last_output_shape: Vec<Shape>,
    pub model_polys: Option<HashMap<PolyId, Vec<Element>>>,
    /// This field is only used in macro layers like MHA
    pub max_poly_len: usize,
}

// compute input shapes for this node
fn compute_node_input_shapes(
    model_input_shapes: &[Shape],
    shapes: &HashMap<NodeId, Vec<Shape>>,
    id: NodeId,
    node: &Node<Element>,
) -> anyhow::Result<Vec<Shape>> {
    node.inputs
        .iter()
        .map(|edge| {
            Ok(if let Some(node_id) = &edge.node {
                let node_shapes = shapes.get(node_id).ok_or(anyhow!(
                    "Node {node_id} not found in set of previous shapes"
                ))?;
                ensure!(
                    edge.index < node_shapes.len(),
                    "Input for node {} is coming from output {} of node {},
                        but this node has only {} outputs",
                    id,
                    edge.index,
                    node_id,
                    node_shapes.len()
                );
                node_shapes[edge.index].clone()
            } else {
                // input node
                ensure!(
                    edge.index < model_input_shapes.len(),
                    "Input for node {} is the input {} of the model,
                        but the model has only {} inputs",
                    id,
                    edge.index,
                    model_input_shapes.len()
                );
                model_input_shapes[edge.index].clone()
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

fn compute_node_shape<E: ExtensionField>(
    mut ctx_aux: ContextAux,
    model_polys: &mut Vec<(NodeId, HashMap<PolyId, MultilinearExtension<E>>)>,
    step_infos: &mut BTreeMap<NodeId, NodeCtx<E>>,
    shapes: &mut HashMap<NodeId, Vec<Shape>>,
    input_shapes: &[Shape],
    id: NodeId,
    node: &Node<Element>,
) -> anyhow::Result<ContextAux> {
    trace!(
        "Context : {}-th layer {}info generation ...",
        id,
        node.operation.describe()
    );
    trace!(
        "Generating context node with id {id}: {:?}",
        node.describe()
    );
    let node_input_shapes = compute_node_input_shapes(input_shapes, shapes, id, node)?;
    ctx_aux.last_output_shape = node_input_shapes;
    let (info, mut new_aux) = node.step_info(id, ctx_aux)?;
    // Retrieve any model polynomials that need to be committed
    if new_aux.model_polys.is_some() {
        model_polys.push((
            id,
            new_aux
                .model_polys
                .as_mut()
                .unwrap()
                .drain()
                .map(|(poly_id, evals)| {
                    let num_vars = ceil_log2(evals.len());
                    (
                        poly_id,
                        MultilinearExtension::<E>::from_evaluations_vec(
                            num_vars,
                            to_base::<E, _>(evals),
                        ),
                    )
                })
                .collect::<HashMap<PolyId, MultilinearExtension<'_, E>>>(),
        ));
    }
    step_infos.insert(
        id,
        NodeCtx {
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            ctx: info,
        },
    );
    shapes.insert(id, new_aux.last_output_shape.clone());
    Ok(new_aux)
}
