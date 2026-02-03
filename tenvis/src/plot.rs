use anyhow::Context;
use log::{error, info};
use std::{collections::HashMap, io::Write, path::Path};
use zkml::{
    graph::Node,
    model::{Model, llm::Driver},
    parser::{gguf::RawGGUF, llm::models::gpt2::GPT2, onnx::FloatOnnxLoader},
    quantization::AbsoluteMax,
    tensor::TensorTypeParam,
};

use crate::Command;

fn parse_llm<P: AsRef<Path>>(filename: P) -> anyhow::Result<Model<f32>> {
    let driver_f32 = Driver::load_from_model(GPT2::new(), &RawGGUF::new(filename), Some(10))
        .context("loading LLM from file")?;
    Ok(driver_f32.model)
}

fn parse_onnx<P: AsRef<Path>>(filename: P) -> anyhow::Result<Model<f32>> {
    let (_, md) = FloatOnnxLoader::new_with_scaling_strategy(
        filename
            .as_ref()
            .to_str()
            .context("unable to convert filename to str")?,
        AbsoluteMax::new(),
    )
    .with_keep_float(true)
    .build()
    .context("building model from file")?;

    Ok(md.float_model.unwrap())
}

fn make_d2<T: TensorTypeParam>(model: &Model<T>) -> String {
    let mut ax = Vec::new();
    enum Element {
        Input(usize),
        Output(usize),
        Block(Vec<String>),
    }
    let mut node2elt = HashMap::new();
    let mut links = Vec::new();

    let g = model.graph();
    for (node_id, node) in g.forward_iter() {
        match node {
            Node::Input(i) => {
                ax.push(Element::Input(*i));
                node2elt.insert(node_id, ax.len() - 1);
            }
            Node::Inner(_) => {
                let parents = g
                    .incomings(node_id)
                    .map(|(_, e)| e.source())
                    .collect::<Vec<_>>();

                let mut merge = parents.len() == 1;
                for (_, in_edge) in g.incomings(node_id) {
                    let parent_id = in_edge.source();
                    if g.outgoings(parent_id).count() > 1
                        || !matches!(ax[node2elt[&parent_id]], Element::Block(_))
                    {
                        merge = false;
                    }
                }

                let label = format!("{node_id}: {}", node.describe());
                if merge {
                    let parent_elt = node2elt[&parents[0]];
                    let Element::Block(items) = ax.get_mut(parent_elt).unwrap() else {
                        unreachable!()
                    };
                    items.push(label);
                    node2elt.insert(node_id, parent_elt);
                } else {
                    ax.push(Element::Block(vec![label]));
                    node2elt.insert(node_id, ax.len() - 1);
                    for (_, e) in g.incomings(node_id) {
                        links.push((node2elt[&e.source()], node2elt[&e.target()]));
                    }
                }
            }
            Node::Output(o) => {
                ax.push(Element::Output(*o));
                node2elt.insert(node_id, ax.len() - 1);
                for (_, e) in g.incomings(node_id) {
                    links.push((node2elt[&e.source()], node2elt[&e.target()]));
                }
            }
        }
    }

    let mut r = String::new();
    for (i, e) in ax.iter().enumerate() {
        match e {
            Element::Input(o) => {
                r.push_str(&format!("elt_{i}: Input {o} {{ shape: circle }}\n"));
            }
            Element::Output(o) => {
                r.push_str(&format!("elt_{i}: Output {o} {{ shape: circle }}\n"));
            }
            Element::Block(items) => {
                if items.len() > 1 {
                    r.push_str(&format!("elt_{i}: \"\" {{\n  direction:left\n"));
                    for (j, item) in items.iter().enumerate() {
                        r.push_str(&format!("  elt_{i}_{j}: \"{item}\"\n"));
                    }
                    r.push_str("}\n\n");
                } else {
                    r.push_str(&format!("elt_{i}: \"{}\"\n", items[0]));
                }
            }
        }
    }

    for (i, e) in ax.iter().enumerate() {
        if let Element::Block(items) = e {
            let mut iter = items.iter().enumerate().peekable();
            while let Some((j, _)) = iter.next() {
                if iter.peek().is_some() {
                    r.push_str(&format!(
                        "elt_{i}.elt_{i}_{j} -> elt_{i}.elt_{i}_{}\n",
                        j + 1
                    ));
                }
            }
        }
    }

    for (i, j) in links.iter() {
        r.push_str(&format!("elt_{i} -> elt_{j}\n"));
    }

    r
}

pub fn plot(cmd: &Command) -> anyhow::Result<()> {
    let Command::Plot {
        model,
        output,
        open,
    } = cmd
    else {
        unreachable!()
    };

    let extension = model
        .extension()
        .context("file name has no extension")?
        .to_str()
        .context("file name is invalid")?;

    let model = match extension {
        "gguf" | "json" => parse_llm(model),
        "onnx" => parse_onnx(model),
        other => anyhow::bail!("unknown model file extension: {other}"),
    }?;

    let d2_script = make_d2(&model);

    if let Some(output) = output {
        info!("generating model graph");
        let extension = output
            .extension()
            .context("target file has no extension")?
            .to_str()
            .context("unable to read extension")?;

        anyhow::ensure!(["svg", "txt"].contains(&extension), "unknown extension");

        let mut script =
            tempfile::NamedTempFile::new().context("failed to create a temporary file")?;
        script
            .write_all(d2_script.as_bytes())
            .context("failed to write to output file")?;
        script.disable_cleanup(true);
        let mut d2_script = script.into_temp_path();
        d2_script.disable_cleanup(true);

        let d2_output = std::process::Command::new("d2")
            .arg(&d2_script)
            .arg(output)
            .output()
            .context("while running `d2`")?;

        if !d2_output.status.success() {
            error!("failed to run d2: {}", d2_output.status);
            info!("STDOUT:");
            std::io::stdout().write_all(&d2_output.stdout)?;
            info!("STDERR:");
            std::io::stderr().write_all(&d2_output.stderr)?;
        }
        if *open {
            open::that(output).context("failed to open the output file")?;
        }

        d2_script.disable_cleanup(false);
    } else {
        println!("{d2_script}");
    }

    Ok(())
}
