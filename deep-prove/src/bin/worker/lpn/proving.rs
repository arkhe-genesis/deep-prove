use super::ModelFetcher;
use anyhow::{Context, anyhow, bail};
use base64::{Engine, prelude::BASE64_STANDARD};
use deep_prove::middleware::{llm, v1, v2};
use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use std::{collections::HashMap, sync::Arc};
use tenstore::GenStore;
use tracing::{info, info_span, warn};
use transcript::BasicTranscript;
use zkml::{
    graph::{
        executor::ThreadPoolExecutor,
        partition::{Partition, PartitionScheduler},
    },
    iop::{context::VerifierContext, prover_graph::ProverGraphIO},
    model::{
        exec_graph::{
            ExecGraphIO, ExecGraphNode, SerializableGraphCtx, SerializablePartitionOutput,
        },
        llm::LLMVerifierContext,
    },
    parser::llm::Token,
};

type F = GoldilocksExt2;
type Pcs = Basefold<F, BasefoldRSParams>;
type T = BasicTranscript<F>;
type ChunkPartition<'a, 'b> = Partition<ExecGraphNode<'a, 'b, F, T, Pcs>, usize>;

/// Execute a chunk job using chunk-based proving with GW mediated coordination.
///
/// This function handles both source partitions with initial inputs and
/// non-source partitions with dependency outputs from other chunks.
///
/// For non-source partitions, the dependency outputs are injected via
/// `set_child_partition_output` before execution can proceed.
#[tracing::instrument(
    name = "run_chunk_partition",
    skip_all,
    fields(plan_id = %chunk.plan_id, chunk_id = chunk.chunk_id, is_source = chunk.is_source)
)]
fn run_chunk_partition(chunk: v2::ChunkPayload, tenstore: GenStore) -> anyhow::Result<Vec<u8>> {
    let ctx_bytes = chunk.ctx.as_bytes();

    info!(
        "run_chunk_partition: starting chunk {} (is_llm={}, ctx_bytes len={}, partition len={})",
        chunk.chunk_id,
        chunk.is_llm(),
        ctx_bytes.len(),
        chunk.partition.len(),
    );

    let ctx: SerializableGraphCtx<F, Pcs> =
        rmp_serde::from_slice(ctx_bytes).context("failed to deserialize SerializableGraphCtx")?;
    info!("run_chunk_partition: deserialized SerializableGraphCtx successfully");

    let mut partitions: Vec<ChunkPartition> = rmp_serde::from_slice(
        &BASE64_STANDARD
            .decode(&chunk.partition)
            .context("failed to decode partition from Base64")?,
    )
    .context("failed to deserialize partitions")?;
    info!(
        "run_chunk_partition: deserialized {} partitions",
        partitions.len()
    );

    // Attach the shared tensor store to partition inputs so that the
    // inference results are stored in a location accessible by other chunks.
    for partition in &mut partitions {
        for (_node_id, input) in partition.inputs.iter_mut() {
            input.attach_store(tenstore.clone());
        }
    }

    if !chunk.is_source && !chunk.dependency_outputs.is_empty() {
        partitions.retain(|p| !p.child_partition.is_empty());
    }

    // Create the partition scheduler with ThreadPoolExecutor for parallel execution
    let tenstore_for_ctx = tenstore.clone();
    let mut scheduler = PartitionScheduler::<_, _, ThreadPoolExecutor>::new(
        partitions,
        ctx.to_full_ctx(tenstore_for_ctx),
        (),
    )?;

    // For non-source partitions, inject dependency outputs before execution
    if !chunk.is_source {
        for (dep_chunk_id, dep_output_b64) in &chunk.dependency_outputs {
            let decoded_bytes = BASE64_STANDARD.decode(dep_output_b64).with_context(|| {
                format!(
                    "failed to decode dependency output from chunk {}",
                    dep_chunk_id
                )
            })?;

            let all_outputs: Vec<SerializablePartitionOutput<F, Pcs>> =
                rmp_serde::from_slice(&decoded_bytes).with_context(|| {
                    format!("failed to deserialize outputs from chunk {}", dep_chunk_id)
                })?;

            let mut dependency_output = all_outputs
                .into_iter()
                .find(|out| out.to == Some(chunk.chunk_id))
                .with_context(|| {
                    format!(
                        "no output from chunk {} destined for chunk {}",
                        dep_chunk_id, chunk.chunk_id
                    )
                })?;

            // Attach the shared tensor store to deserialized outputs
            dependency_output.output.attach_store(tenstore.clone());

            scheduler
                .set_child_output(dependency_output.from, dependency_output.output)
                .with_context(|| {
                    format!("failed to set child output from chunk {}", dep_chunk_id)
                })?;
        }
    }

    // Execute partitions and collect outputs
    // Intermediate outputs (to != None) are destined for other partitions
    // Final outputs (to == None) represent completed computation (proof + IO)
    let mut final_outputs: Vec<ExecGraphIO<F, Pcs>> = Vec::new();
    let mut intermediate_outputs: Vec<SerializablePartitionOutput<F, Pcs>> = Vec::new();

    while !scheduler.is_done() {
        let outputs = scheduler.try_run_partition()?;
        if outputs.is_empty() {
            warn!(
                "partition returned no outputs but is_done=false for chunk {}",
                chunk.chunk_id
            );
            break;
        }

        for output in outputs {
            if output.is_final_output() {
                final_outputs.push(output.output);
            } else {
                intermediate_outputs.push(SerializablePartitionOutput::new(
                    output.from,
                    output.to,
                    output.output,
                ));
            }
        }
    }

    if final_outputs.is_empty() && intermediate_outputs.is_empty() {
        bail!("chunk {} produced no outputs", chunk.chunk_id)
    }

    for output in final_outputs {
        intermediate_outputs.push(SerializablePartitionOutput::new(
            chunk.chunk_id,
            None,
            output,
        ));
    }
    rmp_serde::to_vec(&intermediate_outputs).context("failed to serialize chunk outputs")
}

/// Resolve context into [`v2::ChunkContext`] by fetching from S3 and mmap it.
async fn resolve_context(
    graph_ctx_key: &str,
    fetcher: &ModelFetcher,
    label: &str,
) -> anyhow::Result<v2::ChunkContext> {
    info!(
        "{}: fetching context from S3 for key: {}",
        label, graph_ctx_key
    );
    let mmap = fetcher
        .fetch_graph_context_mmap(graph_ctx_key)
        .await
        .with_context(|| format!("{}: fetching plan context from S3", label))?;
    info!("{}: mmap'd context ({} bytes)", label, mmap.len());
    Ok(v2::ChunkContext::new(Arc::new(mmap)))
}

pub(super) async fn process_job(
    job: v2::GwToWorker,
    tenstore: GenStore,
    fetcher: &ModelFetcher,
) -> anyhow::Result<Vec<u8>> {
    let payload_type = match &job.payload {
        v2::JobPayload::Aggregation(_) => "aggregation",
        v2::JobPayload::Chunk(_) => "chunk",
    };
    info!("process_job: job_id={}, type={}", job.job_id, payload_type);

    match job.payload {
        // Handle aggregation jobs
        v2::JobPayload::Aggregation(ref agg_job) => {
            let span = info_span!("process_aggregation", plan_id = %agg_job.plan_id);
            let _entered = span.enter();

            // Decode and process chunk outputs to extract ChunkProofs and ModelIO
            let mut chunk_outputs_map: HashMap<usize, Vec<SerializablePartitionOutput<F, Pcs>>> =
                HashMap::new();
            let mut model_io: Option<zkml::IO<F>> = None;

            for (i, proof_b64) in agg_job.chunk_proofs.iter().enumerate() {
                let outputs: Vec<SerializablePartitionOutput<F, Pcs>> = rmp_serde::from_slice(
                    &BASE64_STANDARD
                        .decode(proof_b64)
                        .with_context(|| format!("failed to decode chunk proof {}", i))?,
                )
                .with_context(|| format!("failed to deserialize chunk {} outputs", i))?;

                for output in &outputs {
                    // Extract ModelIO from chunk 0's outputs
                    if let ExecGraphIO::ModelIO(io) = &output.output
                        && model_io.is_none()
                    {
                        model_io = Some(io.clone());
                    }
                }

                // Group outputs by their source chunk_id
                for output in outputs {
                    chunk_outputs_map
                        .entry(output.from)
                        .or_default()
                        .push(output);
                }
            }

            let io = model_io.ok_or_else(|| anyhow!("no ModelIO found in chunk outputs"))?;

            let ctx = resolve_context(&agg_job.graph_ctx_key, fetcher, "aggregation").await?;

            let aggregation_partition = agg_job
                .aggregation_partition
                .as_ref()
                .ok_or_else(|| anyhow!("aggregation_partition is required"))?;

            let agg = v2::AggregationPayload::from_job(
                agg_job.clone(),
                ctx,
                aggregation_partition.clone(),
            );

            // Set run_id from plan_id so we use the same tensor store namespace
            let tenstore_for_agg = tenstore.with_run_id(&agg.plan_id);
            let mut dependency_outputs: HashMap<String, String> = HashMap::new();

            // Package ChunkProof outputs from each chunk as dependency_outputs
            for (chunk_id, outputs) in &chunk_outputs_map {
                if *chunk_id != 0 {
                    let mut chunk_proofs: Vec<_> = outputs
                        .iter()
                        .filter(|o| {
                            matches!(&o.output, ExecGraphIO::Prover(ProverGraphIO::ChunkProof(_)))
                                && o.to == Some(0)
                        })
                        .cloned()
                        .collect();

                    // Attach store to each ChunkProof's internal Trace
                    for proof_output in &mut chunk_proofs {
                        proof_output.output.attach_store(tenstore_for_agg.clone());
                    }

                    if !chunk_proofs.is_empty() {
                        let serialized = rmp_serde::to_vec(&chunk_proofs).with_context(|| {
                            format!("failed to serialize chunk {} outputs", chunk_id)
                        })?;
                        dependency_outputs
                            .insert(chunk_id.to_string(), BASE64_STANDARD.encode(&serialized));
                    }
                }
            }

            let chunk_payload = v2::ChunkPayload {
                plan_id: agg.plan_id.clone(),
                chunk_id: 0,
                partition: agg.aggregation_partition.clone(),
                ctx: agg.ctx.clone(),
                dependencies: dependency_outputs
                    .keys()
                    .filter_map(|k| k.parse().ok())
                    .collect(),
                is_source: false,
                dependency_outputs,
                user_tokens: agg.user_tokens.clone(),
                max_context: None,
            };

            let result = run_chunk_partition(chunk_payload, tenstore_for_agg)?;

            let outputs: Vec<SerializablePartitionOutput<F, Pcs>> = rmp_serde::from_slice(&result)
                .context("failed to deserialize chunk 0 aggregation outputs")?;

            let proof = outputs
                .into_iter()
                .find_map(|output| {
                    if let ExecGraphIO::Prover(ProverGraphIO::FinalProof(p)) = output.output {
                        Some(p)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| anyhow!("no FinalProof found in chunk 0 aggregation output"))?;

            if agg.is_llm() {
                let user_tokens: Vec<Token> = agg
                    .user_tokens
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing user_tokens for LLM aggregation"))?
                    .iter()
                    .map(|&t| Token::from(t as u64))
                    .collect();

                let verifier_ctx: LLMVerifierContext<F, Pcs> = rmp_serde::from_slice(
                    &BASE64_STANDARD
                        .decode(&agg.serialized_verifier_ctx)
                        .context("failed to decode LLM verifier context")?,
                )
                .context("failed to deserialize LLM verifier context")?;

                let outputs = vec![v1::LlmOutput {
                    outputs: vec![],
                    proof: llm::LlmProvable {
                        proof,
                        io,
                        ctx: verifier_ctx,
                        user_tokens,
                    },
                }];
                rmp_serde::to_vec(&outputs).context("failed to serialize aggregated LLM proof")
            } else {
                let verifier_ctx: VerifierContext<F, Pcs> = rmp_serde::from_slice(
                    &BASE64_STANDARD
                        .decode(&agg.serialized_verifier_ctx)
                        .context("failed to decode verifier context")?,
                )
                .context("failed to deserialize verifier context")?;

                let outputs = vec![v1::Output {
                    outputs: vec![],
                    proof: v2::Provable {
                        proof,
                        io,
                        ctx: verifier_ctx,
                    },
                }];
                rmp_serde::to_vec(&outputs).context("failed to serialize aggregated proof")
            }
        }

        // Handle chunk jobs
        v2::JobPayload::Chunk(job) => {
            let span = info_span!("process_chunk", plan_id = %job.plan_id, chunk_id = job.chunk_id);
            let _entered = span.enter();

            let ctx = resolve_context(
                &job.graph_ctx_key,
                fetcher,
                &format!("chunk {}", job.chunk_id),
            )
            .await?;

            // Convert wire struct to runtime struct
            let chunk = v2::ChunkPayload::from_job(job, ctx);

            // Set run_id from plan_id so all chunks share the same tensor store namespace
            let tenstore_for_chunk = tenstore.with_run_id(&chunk.plan_id);

            run_chunk_partition(chunk, tenstore_for_chunk)
        }
    }
}
