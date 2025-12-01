use std::rc::Rc;

use crate::{
    Term,
    core::{GlobalContext, Snapshot},
    renderer::{Projection, TensorViewer},
    ui::{Entry, menu},
};
use colored::Colorize;
use dialoguer::FuzzySelect;
use log::error;
use serde::{Deserialize, Serialize};
use tenstore::{GenericStore, StorageKey};
use zkml::{
    Element, NextPowerOfTwo, Number, Tensor, graph::NodeOutput, model::ToStorageKey,
    tensor::TensorTypeParam,
};

enum RunMode {
    F32,
    Element,
}
impl RunMode {
    fn toggle(&mut self) {
        match self {
            RunMode::F32 => *self = RunMode::Element,
            RunMode::Element => *self = RunMode::F32,
        }
    }
}

pub struct Repl {
    term: Term,
    ctx: GlobalContext,
    mode: RunMode,
}

impl Repl {
    pub fn new(ctx: GlobalContext, term: Term) -> Self {
        Repl {
            term,
            ctx,
            mode: RunMode::F32,
        }
    }

    fn ls_tensors(&self) {
        let store = match self.mode {
            RunMode::F32 => self.ctx.snap_f32.store.clone(),
            RunMode::Element => self.ctx.snap_elt.store.clone(),
        };

        println!("{store:?}");
    }

    fn show_model(&self) -> anyhow::Result<()> {
        let settings = crate::model::Settings::default();
        match self.mode {
            RunMode::F32 => crate::model::show_tracks(self.ctx.snap_f32.model.graph(), settings),
            RunMode::Element => {
                crate::model::show_tracks(self.ctx.snap_elt.model.graph(), settings)
            }
        }
    }

    fn show_tensor<T: Number + TensorTypeParam + Serialize + for<'a> Deserialize<'a>>(
        &mut self,
        tty: &mut console::Term,
        snap: Rc<Snapshot<T>>,
    ) -> anyhow::Result<()> {
        let (addrs, pretty_tensors): (Vec<NodeOutput>, Vec<String>) = snap
            .model
            .graph()
            .nodes()
            .flat_map(|(node_id, node)| {
                snap.model
                    .graph()
                    .outgoing_feeds(*node_id)
                    .into_iter()
                    .map(|feed| {
                        let source = feed.source;
                        let min_max = snap
                            .min_max
                            .scaling_range(source)
                            .map(|(min, max)| format!("[{min:.2}; {max:.2}] "))
                            .unwrap_or_default();
                        (
                            source,
                            format!(
                                "{:<15}{source}: {} {}",
                                min_max,
                                snap.shapes[&feed.source.node_id].output_shapes[feed.source.port],
                                node.describe()
                            ),
                        )
                    })
            })
            .unzip();

        let key_id = FuzzySelect::new()
            .with_prompt("tensor")
            .items(&pretty_tensors)
            .clear(true)
            .max_length(30)
            .interact()?;
        let addr = &addrs[key_id];
        let key: StorageKey<Vec<T>> = addr.to_storage_key();
        let data = snap.store.clone().fetch(&key)?;
        let shape = snap.shapes[&addr.node_id].output_shapes[*addr.port].to_owned();
        let tensor = if shape.numel() == data.len() {
            Tensor::new(shape.clone(), data)?
        } else if shape.next_power_of_two().numel() == data.len() {
            Tensor::new(shape.next_power_of_two(), data)?
        } else {
            println!("don't know what to do: {shape:?} vs. {}", data.len());
            return Ok(());
        };
        println!(
            "\n\n\nSelected tensor: {} - {}, {} elts.",
            addr.to_string().blue(),
            format!("{}", shape).bright_white(),
            shape.numel()
        );

        let mut p = Projection::new(tensor.shape().clone());
        let mut tview = TensorViewer::new(&tensor, p.clone());

        #[derive(Clone)]
        enum TViewAction {
            Projection,
            Back,
        }
        let choices = [
            Entry {
                chord: "p".into(),
                label: "project...".into(),
                payload: TViewAction::Projection,
                show: true,
            },
            Entry {
                chord: "q".into(),
                label: "back".into(),
                payload: TViewAction::Back,
                show: true,
            },
        ];

        tview.intro(self.term);
        tview.render(self.term)?;

        #[allow(clippy::while_let_loop)]
        loop {
            match menu(
                tty,
                &format!(
                    "examining tensor {} ({}, {} elts.)",
                    addrs[key_id],
                    tensor.shape(),
                    tensor.shape().numel()
                ),
                &choices,
            )? {
                TViewAction::Projection => {
                    let input = dialoguer::Input::<String>::new()
                        .with_prompt("define projection")
                        .with_initial_text(p.pretty())
                        .interact_text()?;
                    match p.set_from_string(&input) {
                        Ok(_) => {
                            tview.set_projection(p.clone());
                            tview.render(self.term)?;
                        }
                        Err(err) => error!("{err}"),
                    }
                }
                TViewAction::Back => break,
            }
        }

        Ok(())
    }

    pub fn run(mut self, tty: &mut console::Term) -> anyhow::Result<()> {
        #[derive(Clone)]
        enum Action {
            ListTensors,
            ShowTensor,
            ShowModel,
            ToggleMode,
            Quit,
        }

        let root_choices = [
            Entry {
                chord: "m".into(),
                label: "show model".into(),
                payload: Action::ShowModel,
                show: true,
            },
            Entry {
                chord: "l".into(),
                label: "list tensors".into(),
                payload: Action::ListTensors,
                show: true,
            },
            Entry {
                chord: "t".into(),
                label: "show tensor".into(),
                payload: Action::ShowTensor,
                show: true,
            },
            Entry {
                chord: "q".into(),
                label: "quit".into(),
                payload: Action::Quit,
                show: true,
            },
        ];

        loop {
            let choices = root_choices
                .iter()
                .cloned()
                .chain([Entry {
                    chord: "M".into(),
                    label: match self.mode {
                        RunMode::F32 => "mode: [f32] element",
                        RunMode::Element => "mode: f32 [element]",
                    }
                    .into(),
                    payload: Action::ToggleMode,
                    show: true,
                }])
                .collect::<Vec<_>>();
            match menu(tty, "", &choices)? {
                Action::ListTensors => self.ls_tensors(),
                Action::ShowModel => self.show_model()?,
                Action::ToggleMode => self.mode.toggle(),
                Action::ShowTensor => {
                    match self.mode {
                        RunMode::F32 => {
                            self.show_tensor::<f32>(tty, self.ctx.snap_f32.clone())?;
                        }
                        RunMode::Element => {
                            self.show_tensor::<Element>(tty, self.ctx.snap_elt.clone())?;
                        }
                    };
                }
                Action::Quit => break,
            }
        }
        Ok(())
    }
}
