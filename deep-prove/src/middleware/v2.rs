use ff_ext::GoldilocksExt2;
use memmap2::Mmap;
use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use transcript::BasicTranscript;
use zkml::{IO, Proof as ZkmlProof, inputs::Input, iop::context::VerifierContext, verify};

/// The extension field the proving system is based on.
pub type E = GoldilocksExt2;
pub type T = BasicTranscript<E>;

/// A wrapper for a proof and its ancillaries, required by the verifying process.
#[derive(Serialize, Deserialize)]
pub struct Provable {
    pub proof: ZkmlProof<E, Basefold<E, BasefoldRSParams>>,
    pub io: IO<E>,
    pub ctx: VerifierContext<E, Basefold<E, BasefoldRSParams>>,
}
impl Provable {
    pub fn verify(self) -> anyhow::Result<()> {
        verify::<_, T, _>(&self.ctx, self.proof, self.io)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum TaskClass {
    RunOnnx {
        /// The ID of the model to use.
        model_id: i32,

        /// An array of inputs to run proving for
        input: Input,
    },
    RunLlm {
        /// The ID of the model to use.
        model_id: i32,

        /// The prompt text to run inference for.
        prompt: String,

        /// Maximum number of new tokens to generate.
        max_new_tokens: usize,
    },
}

#[derive(Serialize, Deserialize)]
pub struct ClientToGw {
    /// The user-facing name of the submitted task.
    pub pretty_name: String,

    #[serde(flatten)]
    /// The kind of class to run.
    pub class: TaskClass,

    /// The max. cost the user is disposed to pay for the task to be executed.
    pub max_fee: u128,
}

/// Context fetched from S3 and memory-mapped for chunk proving.
#[derive(Clone)]
pub struct ChunkContext(Arc<Mmap>);

impl ChunkContext {
    pub fn new(mmap: Arc<Mmap>) -> Self {
        Self(mmap)
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_ref()
    }
}

/// Chunk proving jobs (wire format) sent by the gateway.
#[derive(Serialize, Deserialize, Clone)]
pub struct ChunkJob {
    pub plan_id: String,
    pub chunk_id: usize,
    pub partition: String,
    /// S3 key to fetch the serialized context from storage.
    pub graph_ctx_key: String,
    /// List of chunk_ids this partition/chunk depends on.
    pub dependencies: Vec<usize>,
    /// Flag to indicate if this is a source partition/chunk.
    pub is_source: bool,
    /// Intermediate outputs from dependent partitions/chunks keyed by chunk_id.
    pub dependency_outputs: HashMap<String, String>,
    /// User input tokens for LLMs.
    pub user_tokens: Option<Vec<usize>>,
    /// Max context window for LLMs.
    pub max_context: Option<usize>,
}

/// Runtime data for chunk proving constructed from [`ChunkJob`] after context
/// resolution or by the aggregation path for synthetic chunk-0.
pub struct ChunkPayload {
    pub plan_id: String,
    pub chunk_id: usize,
    pub partition: String,
    /// Resolved context bytes
    pub ctx: ChunkContext,
    /// List of chunk_ids this partition/chunk depends on.
    pub dependencies: Vec<usize>,
    /// Flag to indicate if this is a source partition/chunk.
    pub is_source: bool,
    /// Intermediate outputs from dependent partitions/chunks keyed by chunk_id.
    pub dependency_outputs: HashMap<String, String>,
    /// User input tokens for LLMs.
    pub user_tokens: Option<Vec<usize>>,
    /// Max context window for LLMs.
    pub max_context: Option<usize>,
}

impl ChunkPayload {
    /// Determine if this chunk is for LLM based on the presence of user input tokens.
    pub fn is_llm(&self) -> bool {
        self.user_tokens.is_some()
    }

    /// Build payload for chunking proving
    pub fn from_job(job: ChunkJob, ctx: ChunkContext) -> Self {
        Self {
            plan_id: job.plan_id,
            chunk_id: job.chunk_id,
            partition: job.partition,
            ctx,
            dependencies: job.dependencies,
            is_source: job.is_source,
            dependency_outputs: job.dependency_outputs,
            user_tokens: job.user_tokens,
            max_context: job.max_context,
        }
    }
}

/// Aggregation jobs (wire format) sent by the gateway
#[derive(Serialize, Deserialize, Clone)]
pub struct AggregationJob {
    pub plan_id: String,
    pub expected_chunks: usize,
    pub chunk_proofs: Vec<String>,
    pub serialized_verifier_ctx: String,
    /// S3 key to fetch the graph context for aggregation.
    pub graph_ctx_key: String,
    /// Partition data for running the aggregation step.
    pub aggregation_partition: Option<String>,
    /// User input tokens for verification (LLM).
    pub user_tokens: Option<Vec<usize>>,
}

/// Runtime data for aggregation proving constructed from [`AggregationJob`]
/// after context resolution and partition validation.
pub struct AggregationPayload {
    pub plan_id: String,
    pub expected_chunks: usize,
    pub chunk_proofs: Vec<String>,
    pub serialized_verifier_ctx: String,
    /// Resolved context bytes
    pub ctx: ChunkContext,
    /// Partition data for running the aggregation step.
    pub aggregation_partition: String,
    /// User input tokens for verification (LLM).
    pub user_tokens: Option<Vec<usize>>,
}

impl AggregationPayload {
    pub fn is_llm(&self) -> bool {
        self.user_tokens.is_some()
    }

    /// Build payload for aggregation job execution
    pub fn from_job(job: AggregationJob, ctx: ChunkContext, aggregation_partition: String) -> Self {
        Self {
            plan_id: job.plan_id,
            expected_chunks: job.expected_chunks,
            chunk_proofs: job.chunk_proofs,
            serialized_verifier_ctx: job.serialized_verifier_ctx,
            ctx,
            aggregation_partition,
            user_tokens: job.user_tokens,
        }
    }
}

/// Job payload for worker execution.
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum JobPayload {
    /// Chunk proving job
    #[serde(rename = "chunk")]
    Chunk(ChunkJob),

    /// Aggregation job
    #[serde(rename = "aggregation")]
    Aggregation(AggregationJob),
}

#[derive(Serialize, Deserialize)]
pub struct GwToWorker {
    /// The job ID to use when communicating with the gateway.
    pub job_id: i64,

    /// The job payload - determines what type of work the worker should do.
    pub payload: JobPayload,
}
