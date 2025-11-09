use std::path::PathBuf;

use crate::{core::GlobalContext, repl::Repl};
use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use log::warn;

mod core;
mod model;
mod plot;
pub(crate) mod renderer;
mod repl;
mod ui;

#[derive(ValueEnum, Debug, Copy, Clone)]
enum Term {
    /// Use ASCII blocks
    Tty,
    /// Use a Qt window
    Qt,
    /// Use the sixel protocol
    Sixel,
    /// Generate a PNG
    Png,
}

#[derive(ValueEnum, Debug, Copy, Clone)]
enum LlmModel {
    Gpt2,
    Gemma3,
}

#[derive(Parser)]
#[command(about, long_about=None)]
struct Args {
    #[arg(short, long, default_value = "qt")]
    term: Term,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a graphical representation of a model file
    Plot {
        /// The file containing the model to plot.
        model: PathBuf,

        /// Where to save the graph, print to stdout if not specified; must end
        /// in .svg .txt
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// If set, open the created file immediately.
        #[arg(short = 'x')]
        open: bool,
    },

    /// Run a model and inspect its tensors.
    Run {
        #[command(subcommand)]
        run: Run,
    },
}

#[derive(Subcommand)]
enum Run {
    /// Run an LLM on the provided prompt
    Llm {
        /// The LLM to run.
        #[clap(value_enum)]
        model: PathBuf,

        /// The initial prompt provided to the LLM.
        #[clap(short, long, default_value = "the sky is")]
        prompt: String,

        /// The maximal context width.
        #[clap(short, long, default_value = "10")]
        context: usize,
    },
    /// Run on the provided ONNX model file.
    Onnx {
        /// The model file to run.
        #[clap(short, long)]
        model: PathBuf,

        /// The input to run the model on.
        #[clap(short, long, default_value = "input.json")]
        input: String,
    },
}

fn run_onnx(run: &Run, term: Term) -> anyhow::Result<()> {
    let Run::Onnx { model, input } = &run else {
        unreachable!()
    };

    let ctx = GlobalContext::from_onnx(model, input)?;
    let repl = Repl::new(ctx, term);
    repl.run(&mut console::Term::stdout())
}

fn run_gguf(run: &Run, term: Term) -> anyhow::Result<()> {
    let Run::Llm {
        model,
        prompt,
        context,
    } = &run
    else {
        unreachable!()
    };

    let ctx = GlobalContext::from_gguf(model, prompt, *context)?;
    let repl = Repl::new(ctx, term);
    repl.run(&mut console::Term::stdout())
}

fn main() -> anyhow::Result<()> {
    stderrlog::new()
        .module(module_path!())
        .verbosity(5)
        .init()
        .context("initializing stderrlog")?;
    let args = Args::parse();

    if let Err(err) = which::which("gnuplot") {
        warn!("`gnuplot` not found: {err:?}, plotting will not work.");
    }

    match args.command {
        Command::Plot { .. } => plot::plot(&args.command),
        Command::Run { run } => match run {
            Run::Llm { .. } => run_gguf(&run, args.term),
            Run::Onnx { .. } => run_onnx(&run, args.term),
        },
    }
}
