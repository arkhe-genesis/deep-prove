#![allow(clippy::print_stdout)]
use anyhow::Context;
use metro::{Event, Metro, RenderingSettings, TrackId};
use std::{cell::OnceCell, collections::HashMap};
use zkml::{
    Shape,
    graph::{Node, NodeId, order_by_in_port},
    layers::provable::OpInfo,
    model::ModelGraph,
    padding::PaddingMode,
    tensor::TensorTypeParam,
};

pub(crate) struct Settings {
    /// Disable graph coloring
    pub no_colors: bool,
    /// Graph horizontal stretching factor
    pub splat: usize,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            no_colors: false,
            splat: 4,
        }
    }
}

#[derive(Debug)]
pub struct ShapeStep {
    pub _input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
}

pub fn shape_steps<T: TensorTypeParam>(
    graph: &ModelGraph<T>,
    unpadded_input_shapes: &[Shape],
) -> anyhow::Result<HashMap<NodeId, ShapeStep>> {
    graph.forward_iter().try_fold(
        HashMap::<NodeId, ShapeStep>::new(),
        |mut shapes, (node_id, node)| {
            match node {
                Node::Inner(layer) => {
                    let un = order_by_in_port(
                        graph
                            .incomings(node_id)
                            .flat_map(|(_, e)| e.feeds())
                            .map(|feed| {
                                let ShapeStep { output_shapes, .. } = shapes
                                    .get(&feed.source().node_id())
                                    .with_context(|| {
                                        format!("fetching shape step for {:?}", feed.source())
                                    })
                                    .unwrap();
                                (*feed.target(), output_shapes[*feed.source().port()].clone())
                            }),
                    )
                    .collect::<Vec<_>>();
                    shapes.insert(
                        node_id,
                        ShapeStep {
                            _input_shapes: un.clone(),
                            output_shapes: layer.output_shapes(&un, PaddingMode::NoPadding)?,
                        },
                    );
                }
                Node::Input(i) => {
                    shapes.insert(
                        node_id,
                        ShapeStep {
                            _input_shapes: vec![],
                            output_shapes: vec![unpadded_input_shapes[*i].clone()],
                        },
                    );
                }
                Node::Output(_) => {}
            }
            Ok(shapes)
        },
    )
}

pub(crate) fn show_tracks<N: TensorTypeParam>(
    model: &ModelGraph<N>,
    settings: Settings,
) -> anyhow::Result<()> {
    // A register of [to -> (from, track)] hanging links
    let mut hanging: HashMap<NodeId, Vec<(NodeId, TrackId)>> = HashMap::new();
    let mut metro = Metro::with_settings(
        RenderingSettings::default()
            .color(!settings.no_colors)
            .splat(settings.splat),
    );
    let mut next_track_id = 0;

    for (current_node_id, current_node) in model.forward_iter() {
        let main_track_id = OnceCell::new();
        if hanging.contains_key(&current_node_id) {
            for (_, track_id) in hanging.remove(&current_node_id).unwrap().into_iter() {
                let main_track_id = main_track_id.get_or_init(|| track_id);
                if track_id != *main_track_id {
                    metro.push(Event::JoinTrack(track_id, *main_track_id));
                }
            }
        } else {
            if next_track_id != 0 {
                metro.push(Event::StartTrack(next_track_id.into()));
            }
            main_track_id.set(next_track_id.into()).unwrap();
            next_track_id += 1;
        }

        let main_track_id = *main_track_id.get().unwrap();
        metro.push(Event::Station(
            main_track_id,
            format!(
                "#{} ← {}\n{}",
                current_node_id,
                model
                    .incoming_feeds(current_node_id)
                    .into_iter()
                    .map(|feed| { format!("{:?}", feed.source()) })
                    .collect::<Vec<_>>()
                    .join(", "),
                current_node.describe()
            )
            .into(),
        ));

        let outgoings = model.outgoing_feeds(current_node_id);
        if outgoings.is_empty() {
            metro.push(Event::StopTrack(main_track_id));
        } else {
            let mut outputs = outgoings.clone();
            outputs.sort_by(|f1, f2| {
                f1.source()
                    .node_id()
                    .cmp(&f2.target().node_id())
                    .then(f1.target().port().cmp(&f2.target().port()))
            });
            outputs.dedup();

            for (i, output) in outputs.into_iter().enumerate() {
                if i == 0 {
                    hanging
                        .entry(output.target().node_id())
                        .or_default()
                        .push((current_node_id, main_track_id));
                } else {
                    metro.push(Event::SplitTrack(
                        main_track_id,
                        TrackId::from(next_track_id),
                    ));
                    hanging
                        .entry(output.target().node_id())
                        .or_default()
                        .push((current_node_id, TrackId::from(next_track_id)));
                    next_track_id += 1;
                };
            }
        }
    }

    println!("{}", metro.to_string().unwrap());

    Ok(())
}
