use crate::{
    Element,
    commit::context::{
        CommitmentProverCtx, CommitmentVerifierCtx, ContextGenerator, GlobalCommitmentCtx, PolyId,
    },
    layers::provable::{Node, NodeCtx, NodeId, OpInfo},
    lookup::context::{LookupContext, TableType},
    model::{Model, ModelCtx, ToIterator},
    tensor::Shape,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct ProverContext<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentProverCtx<'a, E, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
    /// unpadded shape of the first initial input
    pub unpadded_input_shapes: Vec<Shape>,
}

impl<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> ProverContext<'a, E, PCS>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    pub fn write_to_transcript<T: Transcript<E>>(&self, t: &mut T) -> anyhow::Result<()> {
        self.commitment_ctx.write_to_transcript(t)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct VerifierContext<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    E::BaseField: Serialize + DeserializeOwned,
    E: Serialize + DeserializeOwned,
{
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub steps_info: ModelCtx,
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

pub struct ContextIterator<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    commitment_ctx: ContextGenerator<'a, E, PCS>,
    steps_info: ModelCtx,
    lookup: LookupContext,
    unpadded_input_shapes: Vec<Shape>,
}

impl<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Iterator
    for ContextIterator<'a, E, PCS>
{
    type Item = anyhow::Result<(usize, (ProverContext<'a, E, PCS>, VerifierContext<E, PCS>))>;

    fn next(&mut self) -> Option<Self::Item> {
        self.commitment_ctx.next().map(|item| {
            item.map(|(poly_size, (prover_ctx, verifier_ctx))| {
                let prover_ctx = ProverContext {
                    steps_info: self.steps_info.clone(),
                    commitment_ctx: prover_ctx,
                    lookup: self.lookup.clone(),
                    unpadded_input_shapes: self.unpadded_input_shapes.clone(),
                };

                let verifier_ctx = VerifierContext {
                    steps_info: self.steps_info.clone(),
                    commitment_ctx: verifier_ctx,
                    lookup: self.lookup.clone(),
                    unpadded_input_shapes: self.unpadded_input_shapes.clone(),
                };

                (poly_size, (prover_ctx, verifier_ctx))
            })
        })
    }
}

impl Model<Element> {
    /// Helper method employed to build the context data which is independent from the input shape.
    fn build_global_context_data<'b, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
    ) -> anyhow::Result<(ModelCtx, GlobalCommitmentCtx<'b, E, PCS>, LookupContext)> {
        let input_shapes = &self.input_shapes;
        let tables = BTreeSet::new();
        let mut max_poly_len = input_shapes
            .iter()
            .fold(0usize, |acc, shapes| acc.max(shapes.product()));

        let mut ctx_aux = ContextAux {
            tables,
            last_output_shape: input_shapes.clone(),
            model_polys: None,
            max_poly_len,
        };

        let (step_infos, commitment_ctx, lookup) = {
            let mut model_polys = Vec::<(NodeId, HashMap<PolyId, MultilinearExtension<E>>)>::new();
            let mut step_infos = BTreeMap::new();
            let mut shapes: HashMap<NodeId, Vec<Shape>> = HashMap::new();
            debug!("Context : layer info generation ...");
            for (id, node) in self.to_forward_iterator() {
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
                debug!("node: {id}, max_poly_len: {max_poly_len}");
            }
            // Check to see if we use a lookup table alrger than any of the individual polynomials
            ctx_aux.tables.iter().for_each(|table_type| {
                let multiplicity_vars = table_type.multiplicity_poly_vars();
                max_poly_len = max_poly_len.max(1 << multiplicity_vars);
                debug!("table type: {table_type:?}, max_poly_len: {max_poly_len}");
            });

            debug!("Context : lookup generation ...");
            let lookup_ctx = LookupContext::new(&ctx_aux.tables);

            debug!("Context : commitment generating ...");
            let commitment_ctx =
                GlobalCommitmentCtx::<E, PCS>::new(max_poly_len, model_polys, &lookup_ctx)?;
            (step_infos, commitment_ctx, lookup_ctx)
        };

        Ok((ModelCtx { nodes: step_infos }, commitment_ctx, lookup))
    }

    /// Compute the size of the biggest polynomial to be commited for the given `input_shape`
    pub(crate) fn compute_max_poly_size(&self, input_shapes: &[Shape]) -> anyhow::Result<usize> {
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
            let (_, new_aux) = node.step_info(id, ctx_aux)?;
            shapes.insert(id, new_aux.last_output_shape.clone());
            max_poly_len = max_poly_len.max(new_aux.max_poly_len);
            ctx_aux = new_aux;
        }
        Ok(max_poly_len)
    }

    /// Generate the prover and verifier contexts for the input shape embedded in the model
    pub fn generate_contexts<'a, E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>(
        &self,
    ) -> anyhow::Result<(ProverContext<'a, E, PCS>, VerifierContext<E, PCS>)> {
        let (step_info, commitment_ctx, lookup) = self.build_global_context_data()?;
        let (prover_ctx, verifier_ctx) = commitment_ctx.generate_contexts(None)?;

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

    /// Generate the prover and verifier contexts for all the input shapes provided as input. Note that
    /// multiple `input_shapes` could correspond to the same context, as they might yield the same size
    /// for the biggest polynomial that needs to be committed to for this model.
    /// This method already returns the set of contexts covering all the `input_shapes`, without repetitions.
    /// Given an `input_shape`, the corresponding context in the returned `HashMap` can be fetched by getting the
    /// entry in the `HashMap` identified by key `k = self.compute_max_poly_size(input_shape)`
    pub fn generate_contexts_for_input_shapes<
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    >(
        &self,
        input_shapes: Vec<Vec<Shape>>,
    ) -> anyhow::Result<ContextIterator<E, PCS>> {
        debug!("Building global context");
        let (steps_info, commitment_ctx, lookup) = self.build_global_context_data()?;
        debug!("Built global context");
        let max_poly_sizes = input_shapes
            .into_iter()
            .map(|shapes| self.compute_max_poly_size(&shapes))
            .collect::<anyhow::Result<Vec<_>>>()?;
        debug!("Building all contexts");
        let commitment_ctx_iter = commitment_ctx.generate_all_contexts(max_poly_sizes)?;
        Ok(ContextIterator {
            commitment_ctx: commitment_ctx_iter,
            steps_info,
            lookup,
            unpadded_input_shapes: self.unpadded_input_shapes(),
        })
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
    /// THis field is only used in macro layers like MHA
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
                    "Node {} not found in set of previous shapes",
                    node_id
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
    step_infos: &mut BTreeMap<NodeId, NodeCtx>,
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
