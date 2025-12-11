use std::{io::BufReader, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    Element, ScalingFactor,
    quantization::{ModelMetadata, QUANTIZATION_RANGE, Quantize},
};

/// Inputs to the model
#[derive(Clone, Serialize, Deserialize)]
pub struct Input {
    input_data: Vec<Vec<f32>>,
}

impl Input {
    pub fn from_file<P: AsRef<Path>>(p: P) -> anyhow::Result<Self> {
        let inputs: Self = serde_json::from_reader(BufReader::new(
            std::fs::File::open(p.as_ref()).context("opening inputs file")?,
        ))
        .context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }

    pub fn from_reader<R: std::io::Read>(r: R) -> anyhow::Result<Self> {
        let inputs: Self = serde_json::from_reader(r).context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.input_data.len() > 0);
        for (i, input) in self.input_data.iter().enumerate() {
            for x in input.iter() {
                ensure!(
                    QUANTIZATION_RANGE.contains(x),
                    "input #{i}: value `{x:?}` is out of quantization range {QUANTIZATION_RANGE:?}"
                );
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.input_data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_data.is_empty()
    }

    pub fn filter(&self, indices: Option<&Vec<usize>>) -> Result<Self> {
        if let Some(indices) = indices {
            ensure!(
                indices.iter().all(|i| *i < self.input_data.len()),
                "Index {} is out of range (max: {})",
                indices.iter().max().unwrap(),
                self.input_data.len() - 1
            );
            let input_data = indices
                .iter()
                .map(|i| self.input_data[*i].clone())
                .collect();
            Ok(Self { input_data })
        } else {
            Ok(self.clone())
        }
    }

    pub fn as_floats(&self) -> &[Vec<f32>] {
        &self.input_data
    }

    pub fn to_elements(self, md: &ModelMetadata) -> Vec<Vec<Element>> {
        self.quantize(md.input_scaling(0))
    }
}

impl Quantize for Input {
    type Output = Vec<Vec<Element>>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        self.input_data
            .iter()
            .map(|el| el.quantize(scaling))
            .collect()
    }
}

impl std::str::FromStr for Input {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let inputs: Self = serde_json::from_str(s).context("deserializing inputs")?;
        inputs.validate()?;

        Ok(inputs)
    }
}
