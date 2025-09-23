//! Module handling fusion operations for [Nodes](crate::layers::provable::Node) within the Deep Prove framework.
//! The general idea is that given a [Model](crate::model::Model), we want to optimize the computation graph by fusing
//! certain operations together. This is done by specifying the patterns to fuse, and a fusion function that defines
//! how to map to the fused operations.

use crate::{model::Model, number::Number};
use anyhow::Result;
use std::fmt::Debug;

pub mod impls;

/// Trait used to specify a rule for replacing/modifying/inserting [Nodes](crate::layers::provable::Node) in a [`Model`].
pub trait ModelTransform<N: Number>: Debug {
    /// This applies the transformation returning a new model with the changes applied.
    fn apply(&self, model: Model<N>) -> Result<Model<N>>;
}

/// Function that takes a list of [`ModelTransform`] traits and applies them to a [`Model`].
pub fn apply_transformations<N: Number>(
    mut model: Model<N>,
    transforms: Vec<Box<dyn ModelTransform<N>>>,
) -> Result<()> {
    for transform in transforms {
        model = transform.apply(model)?;
    }
    Ok(())
}
