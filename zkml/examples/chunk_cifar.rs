#![allow(clippy::print_stdout)]
use zkml::{
    self, Element, IO, Proof, ProverContext, Tensor,
    inputs::Input,
    iop::{chunking::DefaultChunkingStrategy, context::VerifierContext},
    model::{Model, exec_graph::InferenceEngine},
    parser::onnx::FloatOnnxLoader,
    quantization::InferenceObserver,
    verify,
};
mod common {
    include!("common/mod.rs");
}

use common::{F, Pcs, T};

use crate::common::GraphRuner;

fn build_model<T: std::io::Read>(
    model_data: &[u8],
    inputs: T,
) -> anyhow::Result<(Model<Element>, Vec<Vec<Element>>)> {
    let run_inputs = Input::from_reader(inputs).expect("failed to load inputs");
    let (model, md) =
        FloatOnnxLoader::from_bytes_with_scaling_strategy(model_data, InferenceObserver::new())
            .build()?;
    Ok((model, run_inputs.to_elements(&md)))
}

struct CifarRunner {
    vk: Option<VerifierContext<F, Pcs>>,
}

impl CifarRunner {
    pub fn new() -> Self {
        Self { vk: None }
    }
}

impl GraphRuner for CifarRunner {
    type ChunkingStrategy = DefaultChunkingStrategy;
    fn setup(
        &mut self,
    ) -> anyhow::Result<(ProverContext<F, Pcs>, InferenceEngine, Vec<Tensor<Element>>)> {
        let (model, inputs) = build_model(
            include_bytes!("../assets/scripts/CNN/cnn-cifar-01.onnx"),
            zstd::Decoder::new(&include_bytes!("../assets/scripts/CNN/input.json.zst")[..])
                .expect("failed to parse zstd"),
        )?;
        let input_tensors = model.load_input_flat(inputs)?;
        let (pk, vk) = model.generate_contexts::<F, Pcs>()?;
        self.vk = Some(vk);
        Ok((pk, InferenceEngine::Generic(model), input_tensors))
    }

    fn chunk_strategy(&self) -> Self::ChunkingStrategy {
        DefaultChunkingStrategy::default()
    }

    fn verify_proof(&self, proof: Proof<F, Pcs>, io: IO<F>) -> anyhow::Result<()> {
        verify::<_, T, _>(self.vk.as_ref().unwrap(), proof, io)
    }
}

fn main() -> anyhow::Result<()> {
    let runner = CifarRunner::new();
    common::main_loop(6, runner).unwrap();
    println!("Done");
    Ok(())
}
