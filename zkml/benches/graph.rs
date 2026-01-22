use std::ops::Range;

use zkml::graph::{Graph, Node};

const LINEAR_GRAPH_POWERS_OF_TWO: Range<i32> = 7..10;

/// Plain graph, no additional data in edges or nodes.
type PlainGraph = Graph<(), (), (), ()>;

#[derive(Debug, Copy, Clone)]
enum GraphArgs {
    Linear { size: i32 },
}

fn default_graphs() -> impl Iterator<Item = GraphArgs> {
    LINEAR_GRAPH_POWERS_OF_TWO.map(|pow2| GraphArgs::Linear { size: 1 << pow2 })
}

fn make_plain_graph(graph: GraphArgs) -> PlainGraph {
    match graph {
        GraphArgs::Linear { size } => {
            let mut graph = PlainGraph::new();
            let input = graph.add_input(()).unwrap();
            let mut previous = graph
                .add_consecutive_node(Node::Inner(()), input, ())
                .unwrap();

            for _ in 1..size {
                previous = graph
                    .add_consecutive_node(Node::Inner(()), previous, ())
                    .unwrap();
            }
            let _output = graph
                .add_consecutive_node(Node::Output(()), previous, ())
                .unwrap();
            graph
        }
    }
}

#[divan::bench_group]
mod iter {

    use crate::{GraphArgs, default_graphs, make_plain_graph};

    #[divan::bench(args = default_graphs())]
    fn forward_iter(bencher: divan::Bencher, args: GraphArgs) {
        let graph = make_plain_graph(args);
        bencher.bench(|| {
            // Using count to consume the iterator with low overhead
            graph.forward_iter().count()
        });
    }

    #[divan::bench(args = default_graphs())]
    fn forward_inners(bencher: divan::Bencher, args: GraphArgs) {
        let graph = make_plain_graph(args);
        bencher.bench(|| {
            // Using count to consume the iterator with low overhead
            graph.forward_inners().count()
        });
    }

    #[divan::bench(args = default_graphs())]
    fn backward_iter(bencher: divan::Bencher, args: GraphArgs) {
        let graph = make_plain_graph(args);
        bencher.bench(|| {
            // Using count to consume the iterator with low overhead
            graph.backward_iter().count()
        });
    }
}

fn main() {
    divan::main();
}
