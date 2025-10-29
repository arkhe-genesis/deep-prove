//! This module implements a prover instance that generates proofs completely
//! locally, in a one-shot manner. After a successful proof generation, they are
//! written to a local file.
use std::io::BufWriter;

use anyhow::Context;
use deep_prove::store::MemStore;
use memmap2::Mmap;
use tracing::info;
use zkml::{inputs::Input, quantization::ScalingStrategyKind};

use crate::RunMode;

/// Run the prover once, directly feeding it the required inputs. The proofs are
/// written to a file.
pub async fn run(args: RunMode) -> anyhow::Result<()> {
    let RunMode::OneShot { onnx, inputs } = args else {
        unreachable!()
    };

    crate::setup_logging(false);

    let input = Input::from_file(&inputs).context("loading input")?;
    let model_file = std::fs::File::open(&onnx).context("opening model file")?;
    let model = unsafe { Mmap::map(&model_file) }
        .context("mmap-ing model file")?
        .to_vec();

    let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
    let scaling_input_hash = None;

    let request = crate::DeepProveRequestV1 {
        model,
        input,
        scaling_strategy,
        scaling_input_hash,
    };
    let store = MemStore::default();
    let proofs = crate::run_model_v1(request, store).await?;

    // create a file to write the proofs to
    let file = tempfile::Builder::new()
        .prefix("proof-")
        .suffix(".json")
        .rand_bytes(10)
        .disable_cleanup(true)
        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;

    serde_json::to_writer(BufWriter::new(file), &proofs).context("writing proofs to file")?;

    info!("Successfully generated {} proofs", proofs.len());

    Ok(())
}
