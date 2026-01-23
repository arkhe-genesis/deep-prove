use anyhow::bail;
use clap::{ArgGroup, Parser, ValueEnum, builder::ArgPredicate};
use ff_ext::GoldilocksExt2;
use libc::{RUSAGE_SELF, getrusage, rusage};
use mpcs::{Basefold, BasefoldRSParams};
use tenstore::GenStore;
use timed_core::Output;
use tracing::info;
use tracing_subscriber::EnvFilter;
use zkml::{
    ProverContext,
    measure::{self, Measure},
    model::llm::{Driver, LLMVerifierContext, WithMaxContext},
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::models::{gemma3::Gemma3, gpt2::GPT2},
        safe::RawSafeTensors,
    },
};

type F = GoldilocksExt2;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
type Pcs<E> = Basefold<E, BasefoldRSParams>;

#[derive(Clone, Debug, ValueEnum)]
enum Model {
    GPT2,
    Gemma3,
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::GPT2 => write!(f, "gpt2"),
            Model::Gemma3 => write!(f, "gemma3"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    group(ArgGroup::new("weights").args(["gguf", "hf"]).required(true)),
    group(ArgGroup::new("length").args(["sequence", "max_context"]).required(true)),
)]
struct LLMArgs {
    /// gguf file to load. It can be a local path or a URL to download.
    #[arg(short, long)]
    gguf: Option<String>,

    /// Hugging Face model ID. It MUST be a safetensors model for now. If the model isn't present
    /// in the cache, it will be downloaded.
    #[arg(long)]
    hf: Option<String>,

    /// max context length (in tokens)
    #[arg(long)]
    max_context: Option<usize>,

    /// When specifying a sequence, the model will try each difference sequence length, just generating 2 tokens each time
    /// So for seqlen = n, it will start with a prompt of n-2 tokens, and generates two tokens.
    #[arg(long, value_delimiter = ',')]
    sequence: Vec<usize>,

    /// min user input length (in tokens)
    #[arg(
        long,
        requires = "max_context",
        conflicts_with = "sequence",
        default_value_t = 1
    )]
    min_user_len: usize,

    /// model to use
    #[arg(short, long, value_enum)]
    model: Model,

    /// DEPRECATED: output file name that records individual methods
    #[arg(short, long, default_value_t = {"bench-llm-deprecated.csv".to_string()})]
    output: String,

    /// Benchmark csv output file name
    #[arg(
        long,
        default_value_if("distributed", ArgPredicate::IsPresent, Some("bench_distributed.csv")),
        default_value = "bench.csv" // Optional: what to use if NOT distributed
    )]
    bench: String,

    /// How many rayon threads to use
    /// If not provided, will use the number of logical cores
    /// If 0, will use the number of physical cores
    #[arg(long)]
    num_threads: Option<usize>,

    /// Profile distributed execution
    #[arg(long, default_value_t = false)]
    distributed: bool,
}

const HEADER_MODEL: &str = "model_name";
const HEADER_MODEL_QUANT: &str = "quantization_time";
const HEADER_CONTEXT_GENERATION: &str = "context_generation_time";
const HEADER_MAX_CONTEXT: &str = "max_context";
const HEADER_NUM_THREADS: &str = "num_threads";
const HEADER_MIN_USER_LEN: &str = "min_user_len";
const HEADER_INFERENCE_TIME: &str = "inference_time";
const HEADER_PROOF_SIZE: &str = "proof_size";

fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    timed_core::set_output(Output::CSV("bench-llm.csv".to_string()));

    let args = LLMArgs::parse();

    // either its spceified and if 0 it's the physical cores otherwise what is specified but no more than the logical cores
    let num_threads = if let Some(nt) = args.num_threads {
        if nt == 0 {
            num_cpus::get_physical()
        } else {
            nt.min(num_cpus::get())
        }
    } else {
        num_cpus::get()
    };
    info!("Using {} threads", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let (max_context, sequence) = if let Some(max_context) = args.max_context {
        (max_context, vec![(args.min_user_len, max_context)])
    } else {
        let mut sequence = args.sequence.clone();
        sequence.sort();
        (
            *sequence.last().unwrap(),
            sequence
                .into_iter()
                .map(|s| (s.saturating_sub(2), s))
                .collect(),
        )
    };

    info!(
        "Running with max context {} and user_prompt->max_length: {:?}",
        max_context,
        sequence
            .iter()
            .map(|(s, m)| format!("({}->{})", s, m))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let driver = if let Some(gguf) = args.gguf {
        let model_path = file_cache::from_cache(&gguf)?;
        match args.model {
            Model::GPT2 => {
                Driver::load_from_model(GPT2::new(), &RawGGUF::new(model_path), Some(max_context))?
            }
            _ => bail!("Model {:?} not supported for gguf", args.model),
        }
    } else if let Some(hf) = args.hf {
        let safe = RawSafeTensors::from_hugging_face_cached(&hf)?;
        match args.model {
            Model::GPT2 => Driver::load_from_model(GPT2::new(), &safe, Some(max_context))?,
            Model::Gemma3 => Driver::load_from_model(Gemma3::new(), &safe, Some(max_context))?,
        }
    } else {
        bail!("Either gguf or hf must be provided");
    };

    let mut premeasure = Measure::new()
        .with(HEADER_MODEL, &args.model.to_string())
        .with(HEADER_NUM_THREADS, &num_threads.to_string());
    let (mut driver, _metadata) =
        premeasure.r(HEADER_MODEL_QUANT, || driver.into_provable_llm(None))?;
    let (prover_ctx, mut verifier_ctx): (ProverContext<F, Pcs<F>>, LLMVerifierContext<F, Pcs<F>>) =
        premeasure.r(HEADER_CONTEXT_GENERATION, || driver.context())?;

    for (user_prompt, max_ctx) in sequence {
        // make a new measure for each trial, but always keep the initial measurements for each sample
        measure::set_global(premeasure.clone());

        measure::set(HEADER_MAX_CONTEXT, max_ctx.to_string());
        measure::set(HEADER_MIN_USER_LEN, user_prompt.to_string());

        driver = driver.with_max_context(max_ctx);
        let user_tokens = driver.random_sequence(user_prompt);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = measure::r(HEADER_INFERENCE_TIME, || {
            driver.run_elements(input_tensor, &mut GenStore::default())
        })?;

        let (proof, io) = if args.distributed {
            distributed::run_distributed(trace, &driver, &prover_ctx)?
        } else {
            let io = trace.to_verifier_io()?;
            let peak_rss = peak_rss_bytes();
            let proof = driver.prove(&prover_ctx, trace)?;
            let new_peak_rss = peak_rss_bytes();
            if new_peak_rss > peak_rss {
                // new_peak_rss is the peak memory consumption during proving
                measure::set(
                    "prove_full_memory_peak",
                    (new_peak_rss / 1024 / 1024).to_string(),
                );
            } else {
                // cannot reliably measure peak memory consumption
                measure::set("prove_full_memory_peak", "NaN".to_string());
            }

            (proof, io)
        };

        let proof_size = rmp_serde::to_vec(&proof)?.len();
        measure::set(HEADER_PROOF_SIZE, proof_size);
        verifier_ctx = verifier_ctx.with_max_context(max_ctx);
        verifier_ctx
            .verify(proof, user_tokens, io)
            .expect("invalid proof");
        if !args.distributed {
            measure::post_process(|metrics| {
                let Ok(proof_time) = metrics.get("prove_full").unwrap().parse::<usize>() else {
                    return;
                };
                let Ok(ctx_length) = metrics.get(HEADER_MAX_CONTEXT).unwrap().parse::<usize>()
                else {
                    return;
                };
                let token_per_second = ctx_length as f64 / (proof_time as f64 / 1000.0);
                metrics.insert("token/sec".to_string(), token_per_second.to_string());
            })?;
        }
        measure::to_csv(&args.bench)?;
    }

    Ok(())
}

fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut r: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut r);

        #[cfg(target_os = "linux")]
        {
            (r.ru_maxrss as u64) * 1024
        }

        #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd"))]
        {
            r.ru_maxrss as u64
        }
    }
}

mod distributed {
    use std::collections::HashMap;

    use super::*;
    use anyhow::{anyhow, ensure};
    use tracing::debug;
    use transcript::BasicTranscript;

    use zkml::{
        Element, IO, Proof, Prover,
        graph::{
            executor::{Executor, ThreadPoolExecutor},
            partition::PartitionScheduler,
            scheduler::ExecGraph,
        },
        iop::{
            chunking::LLMChunkingStrategy,
            prover_graph::{LocalProverCtx, ProverGraphIO, ProverGraphNode},
        },
        model::Trace,
    };

    pub type T = BasicTranscript<F>;

    // Type of nodes of the graph to execute
    pub type Node<'a, 'b> = ProverGraphNode<'a, 'b, F, T, Pcs<F>>;

    // Type of execution graph to be partitioned and executed in the workers
    pub type Graph<'a, 'b> = ExecGraph<Node<'a, 'b>, Color>;

    // Color is used to create the partitions, assign different nodes to different workers.
    // It can be usize or any other type such as IP address etc.
    pub type Color = usize;

    /// What a partition scheduler outputs
    pub type PartitionOutput<'a, 'b> = zkml::graph::partition::PartitionOutput<Node<'a, 'b>, Color>;

    const CHUNK_OUTPUT_SIZE: &str = "chunk_output_size";
    const CHUNK_INPUT_SIZE: &str = "chunk_input_size";

    fn run_next_partition<'a, 'b, E: Executor<Node<'a, 'b>, Color>>(
        schedulers: &mut HashMap<Color, PartitionScheduler<Node<'a, 'b>, Color, E>>,
    ) -> anyhow::Result<Option<PartitionOutput<'a, 'b>>> {
        let mut to_be_sent_outputs = Vec::new();
        let mut final_output = None;
        let mut done_schedulers = Vec::new();
        for (color, scheduler) in schedulers.iter_mut() {
            let outputs = scheduler.try_run_partition()?;
            for out in outputs {
                if let Some(to_node) = out.to {
                    let serialized_output = rmp_serde::to_vec(&out)?;
                    // ToDo: measure serialized_output
                    if to_node == 0 {
                        // this is data sent to the coordinator, so we add it to the set of data sent by workers
                        // to the coordinator
                        measure::accumulate_key(
                            CHUNK_OUTPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else if *color == 0 {
                        // this is data sent by the coordinator, so we add it to the set of data sent to the workers
                        measure::accumulate_key(
                            CHUNK_INPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else {
                        unreachable!(
                            "Data is either sent to coordinator or received by coordinator"
                        )
                    };
                    debug!("Node {} sending output to node {}", color, to_node);
                    to_be_sent_outputs.push((to_node, out));
                } else {
                    // we found the final output
                    final_output = Some(out);
                }
                if scheduler.is_done() {
                    done_schedulers.push(*color);
                }
            }
        }

        for (dest_color, out) in to_be_sent_outputs {
            schedulers
                .get_mut(&dest_color)
                .ok_or(anyhow!("Scheduler not found for color {dest_color}"))?
                .set_child_partition_output(out)?
        }

        for color in done_schedulers {
            schedulers.remove(&color);
        }

        Ok(final_output)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn run_distributed(
        full_trace: Trace<Element>,
        driver: &Driver<Element>,
        prover_ctx: &ProverContext<F, Pcs<F>>,
    ) -> anyhow::Result<(Proof<F, Pcs<F>>, IO<F>)> {
        let io = full_trace.to_verifier_io()?;

        let chunks = prover_ctx.split_in_chunks(None, LLMChunkingStrategy)?;
        let graph: Graph = Prover::build_execution_graph(chunks)?;

        let inputs = Prover::graph_inputs(full_trace, &graph)?;

        ensure!(
            inputs.len() == 1,
            "Expected exactly one input node (coordinator split)"
        );

        let flat_inputs =
            inputs
                .into_iter()
                .fold(Vec::new(), |mut ios, (node_input, chunk_prover_io)| {
                    ios.push((node_input.node_id, chunk_prover_io));
                    ios
                });
        let partitions = graph.partition_by_color(flat_inputs)?;

        let mut schedulers = partitions
            .into_iter()
            .map(|(color, partitions)| {
                let ctx = LocalProverCtx::new(prover_ctx, &driver.model);
                Ok((
                    color,
                    PartitionScheduler::<_, _, ThreadPoolExecutor>::new(partitions, ctx, ())?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let mut final_outputs = Vec::new();
        let mut max_peak_rss = 0;
        while !schedulers.is_empty() {
            let peak_rss = peak_rss_bytes();
            if let Some(final_output) = run_next_partition(&mut schedulers)? {
                final_outputs.push(final_output.output)
            };
            let new_peak_rss = peak_rss_bytes();
            max_peak_rss = max_peak_rss.max(new_peak_rss - peak_rss);
        }
        measure::set(
            "prove_full_memory_peak",
            (max_peak_rss / 1024 / 1024).to_string(),
        );

        // Creates channels pairs to communicate with all other nodes
        ensure!(
            final_outputs.len() == 1,
            "Expected 1 outputs for the graph, {} outputs received",
            final_outputs.len()
        );
        let proof = match final_outputs.pop().unwrap() {
            ProverGraphIO::FinalProof(proof) => proof,
            _ => bail!("Invalid output type found after execution of ProverGraph"),
        };
        Ok((proof, io))
    }
}
