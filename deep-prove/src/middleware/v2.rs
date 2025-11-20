use ff_ext::GoldilocksExt2;
use mpcs::{Basefold, BasefoldRSParams};
use serde::{Deserialize, Serialize};
use transcript::BasicTranscript;
use zkml::{
    IO, Proof as ZkmlProof, inputs::Input, iop::context::VerifierContext,
    quantization::ScalingStrategyKind, verify,
};

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

#[derive(Serialize, Deserialize)]
pub struct GwToWorker {
    /// The job ID to use when communicating with the gateway.
    pub job_id: i64,

    /// Object path relative to the bucket root pointing to the uploaded model.
    pub model_path: String,

    /// An array of inputs to run proving for
    pub input: Input,
}
impl GwToWorker {
    pub fn into_request(self, model: Vec<u8>) -> super::v1::DeepProveRequest {
        super::v1::DeepProveRequest {
            model,
            model_file_hash: None,
            input: self.input,
            scaling_strategy: ScalingStrategyKind::AbsoluteMax,
            scaling_input_hash: None,
        }
    }
}
