//! File containing code for lookup witness generation.
use crate::{
    Element,
    graph::{
        Graph, NodeId, NodeInput,
        executor::{Executor, SequentialExecutor},
        scheduler::{Colored, ExecNode, GraphScheduler},
    },
    iop::{ChallengeStorage, context::ProverContext, prover::ModelLayersRef},
    layers::provable::ProvableOp,
    lookup::{
        logup_gkr::structs::{LogUpInput, LogUpVerifierInstance, ProofType},
        table::Table,
    },
    measure,
    model::Trace,
};
use anyhow::{Context as CC, anyhow, bail};
use ceno_p3::field::{Field, FieldAlgebra};
use ff_ext::ExtensionField;
use itertools::Itertools;
use mpcs::PolynomialCommitmentScheme;
use multilinear_extensions::util::transpose;
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap, btree_map},
    marker::PhantomData,
    sync::Arc,
};
use tracing::{debug, warn};
use transcript::Transcript;
use utils::Metrics;
use witness::{InstancePaddingStrategy, RowMajorMatrix};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    /// Store the tables found in the model, with a list of the nodes
    /// using the given table
    pub(crate) tables: BTreeMap<Table, Vec<NodeId>>,
}

impl LookupContext {
    pub fn new(set: &BTreeMap<Table, Vec<NodeId>>) -> LookupContext {
        LookupContext {
            tables: set.clone(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Table> {
        self.tables.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn create_logup_inputs<PCS, E>(
        &self,
        multiplicities_commitment: &PCS::CommitmentWithWitness,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<Vec<LogUpInput<E>>>
    where
        E: ExtensionField,
        PCS: PolynomialCommitmentScheme<E>,
    {
        // First we retrieve the multiplicity polynomials
        let multiplicity_polys =
            PCS::get_arc_mle_witness_from_commitment(multiplicities_commitment);
        self.iter()
            .zip(multiplicity_polys.iter())
            .map(|(table, m_poly)| {
                let multiplicities = m_poly.get_base_field_vec().to_vec();
                let column_evals = table.get_table_columns::<E>();
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot generate LogUp input",
                        table.name()
                    ))?;
                LogUpInput::<E>::new_table(
                    column_evals,
                    multiplicities,
                    constant_challenge,
                    column_separation_challenge,
                )
                .map_err(|e| anyhow!("Table: {}, {e:?}", table.name()))
            })
            .collect::<anyhow::Result<Vec<LogUpInput<E>>>>()
    }

    pub fn create_logup_verifier_instances<E>(
        &self,
        challenge_storage: &ChallengeStorage<E>,
    ) -> anyhow::Result<Vec<LogUpVerifierInstance<E>>>
    where
        E: ExtensionField,
    {
        self.iter()
            .map(|table| {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot generate LogUp input",
                        table.name()
                    ))?;
                Ok(LogUpVerifierInstance::<E>::new(
                    constant_challenge,
                    column_separation_challenge,
                    table.num_columns(),
                    ProofType::Table,
                    table.table_bit_size() - 1,
                ))
            })
            .collect::<anyhow::Result<Vec<LogUpVerifierInstance<E>>>>()
    }
}

pub(crate) fn count_elements<I: IntoIterator<Item = Element>>(i: I) -> HashMap<Element, u64> {
    let mut count = HashMap::<Element, u64>::new();
    for v in i.into_iter() {
        *count.entry(v).or_default() += 1;
    }
    count
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub struct LookupWitnessGen<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>>
where
    PCS::CommitmentWithWitness: Serialize + for<'a> Deserialize<'a>,
{
    /// Contains the count of elements per table type.
    ///
    /// These values are later used to compute the GKR's multiplicities.
    element_count: BTreeMap<Table, HashMap<Element, u64>>,
    logup_witnesses: HashMap<NodeId, Arc<PCS::CommitmentWithWitness>>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Clone for LookupWitnessGen<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + for<'a> Deserialize<'a>,
{
    fn clone(&self) -> Self {
        LookupWitnessGen {
            element_count: self.element_count.clone(),
            logup_witnesses: self.logup_witnesses.clone(),
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Default for LookupWitnessGen<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        LookupWitnessGen {
            element_count: BTreeMap::default(),
            logup_witnesses: HashMap::default(),
        }
    }
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> LookupWitnessGen<E, PCS>
where
    PCS::CommitmentWithWitness: Serialize + for<'a> Deserialize<'a>,
{
    pub fn insert_logup_witness(&mut self, node_id: NodeId, witness: PCS::CommitmentWithWitness) {
        self.logup_witnesses.insert(node_id, Arc::new(witness));
    }
    pub fn insert_element_count(&mut self, table: Table, elements: HashMap<Element, u64>) {
        self.element_count.insert(table, elements);
    }
    pub fn insert_layer_witness_data(
        &mut self,
        node_id: NodeId,
        witness: PCS::CommitmentWithWitness,
        tables: Vec<Table>,
        element_counts: Vec<HashMap<Element, u64>>,
    ) {
        // First insert the witness
        self.insert_logup_witness(node_id, witness);
        // Now insert each of the lookup counts for their respective tables
        for (table, count) in tables.into_iter().zip(element_counts.into_iter()) {
            if !count.is_empty() {
                self.insert_element_count(table, count);
            }
        }
    }

    /// Consume the lookups and witness of `other` into this instance.
    fn consume(&mut self, other: Self) {
        for (table, elements) in other.element_count.into_iter() {
            match self.element_count.entry(table) {
                btree_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(elements);
                }
                btree_map::Entry::Occupied(mut occupied_entry) => {
                    let agg_count = occupied_entry.get_mut();
                    for (element, count) in elements.into_iter() {
                        *agg_count.entry(element).or_default() += count;
                    }
                }
            }
        }
        self.logup_witnesses.extend(other.logup_witnesses);
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1 << 32;

/// Action to put inside the graph of tasks. This operation can be serialized over the network.
#[derive(Debug, Clone)]
pub(crate) struct GenerateWitness<
    'a,
    'b,
    'c,
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
> {
    _phantom: PhantomData<(E, PCS)>,
    _marker: PhantomData<(&'a (), &'b (), &'c ())>,
}

impl<'a, 'b, 'c, E: ExtensionField, PCS: PolynomialCommitmentScheme<E> + Send + Sync> Default
    for GenerateWitness<'a, 'b, 'c, E, PCS>
{
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
            _marker: PhantomData,
        }
    }
}

pub(crate) struct GenerateWitnessContext<'a, 'b, 'c, E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
{
    trace: &'a Trace<Element>,
    ctx: &'b ProverContext<E, PCS>,
    layers: &'c ModelLayersRef<'c>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: DeserializeOwned"))]
pub(crate) enum GenerateWitnessIO<E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E>,
    PCS::CommitmentWithWitness: Serialize + for<'a> Deserialize<'a>,
{
    Input(NodeId),
    Output(LookupWitnessGen<E, PCS>),
}

impl<'a, 'b, 'c, E, PCS> ExecNode for GenerateWitness<'a, 'b, 'c, E, PCS>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync + 'b,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    type IO = GenerateWitnessIO<E, PCS>;
    type Context = GenerateWitnessContext<'a, 'b, 'c, E, PCS>;

    fn run(&self, ctx: &Self::Context, input: Vec<Self::IO>) -> anyhow::Result<Vec<Self::IO>> {
        let input = input.first().context("expect only one node_id as input")?;
        let GenerateWitnessIO::Input(node_id) = input else {
            bail!("Expected input to be a node_id");
        };

        let step = ctx
            .trace
            .get_step(node_id)
            .with_context(|| format!("fetching trace for {node_id}"))?;
        let op = ctx
            .layers
            .get(node_id)
            .ok_or(anyhow!("Node {node_id} not found in model"))?;
        Ok(vec![
            op.gen_lookup_witness(*node_id, ctx.ctx, step)
                .map_err(|e| {
                    anyhow!(
                        "Error generating lookup witness for node {node_id:?} with error: {e:?}"
                    )
                })
                .map(|gen_w| GenerateWitnessIO::Output(gen_w))?,
        ])
    }
    fn describe(&self) -> String {
        "GenerateWitness".to_string()
    }
}

#[derive(Debug)]
pub struct LookupWitness<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> {
    pub logup_witnesses: HashMap<NodeId, PCS::CommitmentWithWitness>,
    pub table_witnesses: Option<PCS::CommitmentWithWitness>,
}

impl<E: ExtensionField, PCS: PolynomialCommitmentScheme<E>> Default for LookupWitness<E, PCS> {
    fn default() -> Self {
        LookupWitness {
            logup_witnesses: HashMap::default(),
            table_witnesses: None,
        }
    }
}

pub(crate) fn generate_lookup_witness_for_chunk<'a, 'b, 'c, E, T, PCS, N, Ex>(
    chunk_graph: &Graph<N, usize, usize, ()>,
    chunk_lookup_ctx: &LookupContext,
    chunk_trace: &'a Trace<Element>,
    ctx: &'b ProverContext<E, PCS>,
    transcript: &mut T,
    layers: &'c ModelLayersRef<'c>,
    executor_config: &Ex::Config,
) -> anyhow::Result<LookupWitness<E, PCS>>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
    T: Transcript<E>,
    Ex: Executor<GenerateWitness<'a, 'b, 'c, E, PCS>, usize>,
{
    // If the lookup context is empty then there are no lookup witnesses to generate so we return default values
    if chunk_lookup_ctx.is_empty() {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok(LookupWitness::default());
    }

    // Make the witness gen struct that stores relevant table lookup data
    debug!("== Witness poly commitments generation ==");
    let metrics = Metrics::new();
    let mut witness_gen = LookupWitnessGen::<E, PCS>::default();

    // We create the graph here for showcasing the graph module. The end goal is
    // that we create the graph at the top level and every functionality of the
    // prover is appended to that graph
    //
    // We also spin up a local executor here, since this is for the first PR to
    // showcase the graph module as well. The endgoal is that once the full
    // graph is created, the executor will simply run the graph to completion.
    // Since we haven't "graphized" the whole prover yet, we limit the scope to
    // this small function.

    // the colour for now doesn't matter too much since everything is
    // sequential. later on, the local executor can use a threadpool to run the
    // graph in parallel with a master thread
    let max_colour = 2;
    let mut graph = Graph::new();
    let inputs = chunk_graph
        .forward_inners()
        .enumerate()
        .map(|(idx, (node_id, _))| {
            let node_idx =
                graph.add_inner(Colored::new(GenerateWitness::default(), idx % max_colour))?;
            let input = GenerateWitnessIO::Input(node_id);
            Ok((NodeInput::new(node_idx, 0), input))
        })
        .collect::<anyhow::Result<HashMap<NodeInput, GenerateWitnessIO<_, _>>>>()?;

    // here for the moment there is not yet a "parent node" so it's a directed
    // graph ... but with no edges.
    let graph_ctx = GenerateWitnessContext {
        ctx,
        layers,
        trace: chunk_trace,
    };
    let scheduler = GraphScheduler::<GenerateWitness<E, PCS>, usize>::new(graph);
    for gen_w in Ex::run(executor_config, scheduler, inputs, &graph_ctx)?.into_values() {
        let GenerateWitnessIO::Output(gen_w) = gen_w else {
            return Err(anyhow!("Expected output to be a logup witness"));
        };
        witness_gen.consume(gen_w);
    }

    debug!(
        "== Witness poly commitments generation metrics {} ==",
        metrics.to_span()
    );

    debug!("== Witness table multiplicities commitment generation ==");
    let metrics = Metrics::new();

    // calculate the table multiplicities
    let multiplicities = witness_gen
        .element_count
        .par_iter()
        .map(|(table, table_lookup_data)| {
            let table_column = table.get_merged_table_columns();

            // Check to see that all the lookup values are present in the table
            #[cfg(test)]
            {
                let mut total_not_in_table = 0;
                for key in table_lookup_data.keys() {
                    let check = table_column.contains(key);
                    if !check {
                        total_not_in_table += 1;
                    }
                }
                if total_not_in_table > 0 {
                    println!(
                        "For table {}, total lookup values: {}, not in table: {}",
                        table.name(),
                        table_lookup_data.len(),
                        total_not_in_table
                    );
                }
            }
            // We have to account for repeated entries in the lookup table. This is usually the case if the table we want to lookup from is not a power of two size, in that case we pick a row from the table
            // and repeat it until the table has the desired size.
            let table_column_map =
                table_column
                    .iter()
                    .fold(BTreeMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        let table_count = *table_column_map.get(table_val).unwrap();
                        let inv = if table_count != 1 {
                            E::BaseField::from_canonical_u64(table_count).inverse()
                        } else {
                            E::BaseField::ONE
                        };
                        E::BaseField::from_canonical_u64(*lookup_count) * inv
                    } else {
                        E::BaseField::ZERO
                    }
                })
                .collect::<Vec<E::BaseField>>();

            Ok(multiplicities)
        })
        .collect::<anyhow::Result<Vec<Vec<E::BaseField>>>>()?;

    let grouped_by_vars = witness_gen
        .element_count
        .keys()
        .map(|table| (table.table_bit_size(), *table))
        .into_group_map();
    let rmms = grouped_by_vars
        .into_iter()
        .sorted_by(|a, b| Ord::cmp(&b.0, &a.0))
        .scan(0, |skip, (nv, list)| {
            let multiplicities_slice = multiplicities[*skip..*skip + list.len()].to_vec();
            *skip += list.len();
            Some((nv, multiplicities_slice))
        })
        .fold(vec![], |mut acc, (_, multiplicities_slice)| {
            let num_tables = multiplicities_slice.len();
            let transposed = transpose(multiplicities_slice);

            let rmm = RowMajorMatrix::new_by_inner_matrix(
                ceno_p3::matrix::dense::DenseMatrix::new(transposed.concat(), num_tables),
                InstancePaddingStrategy::Default,
            );
            acc.push(rmm);
            acc
        });

    let table_witness = ctx.commitment_ctx.batch_commit(rmms)?;
    debug!(
        "== Witness table multiplicities commitment metrics {} ==",
        metrics.to_span()
    );

    // Write the witness commitments to the transcript
    for (node_id, _) in chunk_graph.forward_iter() {
        if let Some(prover_commit) = witness_gen.logup_witnesses.get(&node_id) {
            let comm = PCS::get_pure_commitment(prover_commit);
            PCS::write_commitment(&comm, transcript).map_err(|e| anyhow!("{e:?}"))?;
        }
    }

    let table_comm = PCS::get_pure_commitment(&table_witness);
    PCS::write_commitment(&table_comm, transcript).map_err(|e| anyhow!("{e:?}"))?;

    Ok(LookupWitness {
        logup_witnesses: witness_gen
            .logup_witnesses
            .into_iter()
            .map(|(k, v)| (k, Arc::into_inner(v).unwrap()))
            .collect(),
        table_witnesses: Some(table_witness),
    })
}

pub fn generate_lookup_witnesses<'a, E, T: Transcript<E>, PCS>(
    trace: &Trace<Element>,
    ctx: &ProverContext<E, PCS>,
    transcript: &mut T,
    layers: &ModelLayersRef<'a>,
) -> anyhow::Result<LookupWitness<E, PCS>>
where
    E: ExtensionField,
    PCS: PolynomialCommitmentScheme<E> + Send + Sync,
    PCS::CommitmentWithWitness: Serialize + DeserializeOwned + Send + Sync,
    PCS::ProverParam: Send + Sync,
{
    measure::r("witness_commitment", || {
        generate_lookup_witness_for_chunk::<_, _, _, _, SequentialExecutor>(
            &ctx.model_ctx.nodes,
            &ctx.lookup,
            trace,
            ctx,
            transcript,
            layers,
            &(),
        )
    })
}
