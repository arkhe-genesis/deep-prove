//! Graph-based computation framework for parallel and distributed execution.
//!
//! This module provides a comprehensive graph-based computation framework that can represent
//! any computation as a directed graph with port-based connections. The framework supports:
//!
//! - **Parallel Execution**: Nodes can be executed concurrently when their dependencies are satisfied
//! - **Distributed Execution**: Graphs can be partitioned and executed across multiple machines
//! - **Flexible Scheduling**: Different scheduling policies for optimal resource utilization
//! - **Port-based Connections**: Fine-grained control over data flow between computation nodes
//!
//! The primary use case is representing proving logic for Interactive Oracle Proofs (IOPs)
//! where different proof components can be computed in parallel or distributed across a network.
//!
//! # Core Components
//!
//! - [`Graph`]: The main graph data structure with nodes and port-based edges
//! - [`scheduler`]: Scheduling logic for determining execution order and readiness
//! - [`executor`]: Execution engines for running graphs (sequential, parallel, etc.)
//! - [`partition`]: Graph partitioning for distributed execution
//!
//! # Example
//!
//! ```rust
//! use zkml::graph::{Graph, scheduler::GraphScheduler, executor::SequentialExecutor};
//!
//! // This example would require implementing ExecNode for your computation type
//! // See the scheduler module for complete examples
//! ```
/// Execution engines for running computational graphs.
///
/// Provides different execution strategies including sequential execution
/// and parallel execution using thread pools.
pub mod executor;

/// Core graph data structure and operations.
///
/// Contains the main [`Graph`] type and all related data structures for
/// representing port-based directed graphs.
#[allow(clippy::module_inception)]
pub mod graph;

/// Graph partitioning for distributed execution.
///
/// Provides functionality to split graphs into independent partitions
/// that can be executed on different machines or processes.
pub mod partition;

/// Scheduling logic for graph execution.
///
/// Contains the [`GraphScheduler`] and related types for determining
/// when nodes are ready to execute and managing execution dependencies.
pub mod scheduler;

// everything directly graph related export as top level
pub use graph::*;
// pub use port::*;

/// Utility macro for extracting specific enum variants from vectors.
///
/// This macro is particularly useful when working with graph nodes that are represented
/// as enum variants. It allows safe extraction of all instances of a specific variant
/// from a vector, returning an error if any element doesn't match the expected variant.
///
/// # Examples
///
/// ```
/// use anyhow::anyhow;
///
/// macro_rules! try_extract_variant_vec {
///     ($variant:ident :: $name:ident ( $inner:ident ), $vec:expr) => {{
///         let mut out: Vec<$inner> = Vec::with_capacity($vec.len());
///         let mut err_i: usize = usize::MAX;
///         for (i, e) in $vec.into_iter().enumerate() {
///             match e {
///                 $variant::$name(inner) => out.push(inner),
///                 _ => {
///                     err_i = i;
///                     break;
///                 }
///             }
///         }
///         if err_i != usize::MAX {
///             Err(anyhow!("Type mismatch at index {}", err_i))
///         } else {
///             Ok(out)
///         }
///     }};
///     ($variant:ident :: $name:ident, $vec:expr) => {{
///         let mut out: Vec<()> = Vec::with_capacity($vec.len());
///         let mut err_i: usize = usize::MAX;
///         for (i, e) in $vec.into_iter().enumerate() {
///             match e {
///                 $variant::$name => out.push(()),
///                 _ => {
///                     err_i = i;
///                     break;
///                 }
///             }
///         }
///         if err_i != usize::MAX {
///             Err(anyhow!("Type mismatch at index {}", err_i))
///         } else {
///             Ok(out)
///         }
///     }};
/// }
///
/// #[derive(Debug, PartialEq)]
/// enum Operation {
///     Add(i32),
///     Multiply(f64),
///     Clear,
/// }
///
/// // Extract variants with payload
/// let ops = vec![Operation::Add(1), Operation::Add(2), Operation::Add(3)];
/// let values = try_extract_variant_vec!(Operation::Add(i32), ops).unwrap();
/// assert_eq!(values, vec![1, 2, 3]);
///
/// // Extract variants without payload
/// let ops2 = vec![Operation::Clear, Operation::Clear];
/// let values2 = try_extract_variant_vec!(Operation::Clear, ops2).unwrap();
/// assert_eq!(values2, vec![(), ()]);
///
/// // Error case - type mismatch
/// let ops3 = vec![Operation::Add(1), Operation::Multiply(2.0)];
/// assert!(try_extract_variant_vec!(Operation::Add(i32), ops3).is_err());
/// ```
#[allow(unused_macros)]
#[macro_export]
macro_rules! try_extract_variant_vec {
    // case: variant with payload
    ($variant:ident :: $name:ident ( $inner:ident ), $vec:expr) => {{
        let mut out: Vec<$inner> = Vec::with_capacity($vec.len());
        let mut err_i: usize = usize::MAX;
        for (i, e) in $vec.into_iter().enumerate() {
            match e {
                $variant::$name(inner) => out.push(inner),
                _ => {
                    println!("Type mismatch {:?}", e);
                    err_i = i;
                    break;
                }
            }
        }
        if err_i != usize::MAX {
            Err(anyhow::anyhow!("Type mismatch at index {}", err_i))
        } else {
            Ok(out)
        }
    }};
    // case: variant without payload
    ($variant:ident :: $name:ident, $vec:expr) => {{
        let mut out: Vec<()> = Vec::with_capacity($vec.len());
        let mut err_i: usize = usize::MAX;
        for (i, e) in $vec.into_iter().enumerate() {
            match e {
                $variant::$name => out.push(()),
                _ => {
                    println!("Type mismatch {:?}", e);
                    err_i = i;
                    break;
                }
            }
        }
        if err_i != usize::MAX {
            Err(anyhow::anyhow!("Type mismatch at index {}", err_i))
        } else {
            Ok(out)
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::graph::scheduler::{ExecGraph, ExecNode, IntoColor};

    use super::*;

    #[derive(Debug, Clone)]
    pub enum TestOperation {
        Test1,
        Test2,
    }

    impl ExecNode for TestOperation {
        type IO = String;
        type Context = ();
        fn describe(&self) -> String {
            match self {
                TestOperation::Test1 => "Test1".to_string(),
                TestOperation::Test2 => "Test2".to_string(),
            }
        }
        fn run(&self, _ctx: &Self::Context, input: Vec<Self::IO>) -> anyhow::Result<Self::IO> {
            match self {
                TestOperation::Test1 => {
                    println!("Test1: {input:?}");
                    Ok(format!("Test1: {input:?}"))
                }
                TestOperation::Test2 => {
                    println!("Test2: {input:?}");
                    Ok(format!("Test2: {input:?}"))
                }
            }
        }
    }

    /// Creates a simple test graph with colored nodes for testing purposes.
    ///
    /// This function creates a basic computational graph where nodes are assigned
    /// colors in a round-robin fashion. The coloring determines which nodes can
    /// be executed on the same machine or thread pool.
    ///
    /// # Parameters
    ///
    /// * `num_colors` - The number of different colors to cycle through
    ///
    /// # Returns
    ///
    /// An executable graph with test operations that can be used for testing
    /// the scheduling and execution framework.
    ///
    /// # Note
    ///
    /// This is a simple implementation that doesn't consider node dependencies
    /// when assigning colors. A more sophisticated implementation would assign
    /// colors based on the dependency graph to optimize parallel execution.
    pub fn instantiate(num_colors: usize) -> ExecGraph<TestOperation, usize> {
        let mut graph = Graph::new();
        let mut color = 0;
        let test1_node = graph.add_node(TestOperation::Test1.colored(color)).unwrap();
        graph.set_input(test1_node, 0, None).unwrap();

        color = (color + 1) % num_colors;
        let test2_node = graph.add_node(TestOperation::Test2.colored(color)).unwrap();
        graph
            .add_edge(test1_node, test2_node, Ports::consecutive(), None)
            .unwrap();
        graph
    }

    #[test]
    fn test_try_extract_variant_vec() {
        #[derive(Debug)]
        enum MyEnum {
            Variant1(i32),
            Variant2(f64),
        }

        let vec = vec![MyEnum::Variant1(1), MyEnum::Variant1(2)];
        let out = try_extract_variant_vec!(MyEnum::Variant1(i32), vec).unwrap();
        assert_eq!(out, vec![1, 2]);

        let vec = vec![MyEnum::Variant2(1.0), MyEnum::Variant2(2.0)];
        let out = try_extract_variant_vec!(MyEnum::Variant2(f64), vec).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);

        let vec = vec![MyEnum::Variant1(1), MyEnum::Variant2(2.0)];
        assert!(try_extract_variant_vec!(MyEnum::Variant1(i32), vec).is_err());
    }
}
