//! An alternative to utils::stream_metrics which produces csv which are complicated to parse and does
//! not support putting custom values. This module is supposed to be simpler and more flexible.
//! You can create first a `[Measure]` object that is meant to record a bunch of timings and other static values _once_
//! Then use the `r()` method to record metrics for any function call.
//! At the end, you can call the `to_csv` method to save the metrics to a csv file.
//! Note that for simplicity, the keys are "dynamic", you don't need to specify the keys in advance, it just
//! records the keys by order of measurement. When calling `to_csv`, it will _append_ to the file so the keys must
//! be the same as the ones in the first row.
//! If you need to make _multiple_ times the same measurement (e.g. have multiple lines per row), don't forget
//! to use a new `[Measure]` object.
//! To use it accross a global codebase without passing the metrics object around, you can use the `set_global_metrics` function.
//! ```rust
//! use zkml::measure::Measure;
//!
//! {
//!     let mut m = Measure::new().with("param_size", "1000");
//!     m.r("sort", || {
//!         let mut vec = vec![1, 2, 3, 4, 5];
//!         vec.sort();
//!         vec
//!     });
//!     m.set("", "1000");
//! }
//! ```
//! or the global variant:
//! ```rust
//! use zkml::measure::{self, Measure};
//!
//! {
//!     measure::set_global(Measure::new().with("param_size", "1000"));
//!     measure::r("sort", || {
//!         let mut vec = vec![1, 2, 3, 4, 5];
//!         vec.sort();
//!         vec
//!     });
//!     measure::to_csv("metrics.csv").unwrap();
//! }
//! ```
//!
//! The global API is recursion friendly, you can call `r` inside another `r` call:
//! ```rust
//! use zkml::measure::{self, Measure};
//!
//! {
//!     let m = Measure::new().with("param_size", "1000");
//!     measure::set_global(m);
//!     measure::r("sort", || {
//!         let mut vec = vec![1, 2, 3, 4, 5];
//!         vec.sort();
//!         measure::r("sort2", || {
//!             let mut vec = vec![1, 2, 3, 4, 5];
//!             vec.sort();
//!         });
//!     });
//! }
//! ```
//!
//!
//! TODO: add the memory metrics as well to get on par with the utils::stream_metrics.
use std::{
    collections::BTreeMap,
    fmt::Display,
    fs::{File, OpenOptions},
    io::BufRead,
    str::FromStr,
    sync::Mutex,
    time::{self, Duration},
};

use anyhow::ensure;
use tracing::trace;

pub static MEASURE: Mutex<Option<Measure>> = Mutex::new(None);

pub fn set_global(metrics: Measure) {
    *MEASURE.lock().unwrap() = Some(metrics);
}

fn r_and_accumulate<R, F: FnOnce() -> R, A: Fn(u128, u128) -> u128>(
    key: &str,
    f: F,
    acc_fn: Option<A>,
) -> anyhow::Result<R> {
    let (result, elapsed) = timeit(f);
    Ok(match MEASURE.lock().unwrap().as_mut() {
        Some(metrics) => {
            let elapsed = elapsed.as_millis();
            if let Some(acc_fn) = acc_fn {
                metrics.accumulate_key(key, elapsed, acc_fn)?
            } else {
                metrics.data.insert(key.to_string(), elapsed.to_string());
            };
            result
        }
        None => {
            trace!("Metrics are not initialised, skipping {key}");
            result
        }
    })
}

pub fn record_timing(key: &str, elapsed: Duration) {
    match MEASURE.lock().unwrap().as_mut() {
        Some(metrics) => {
            metrics.record_timing(key, elapsed);
        }
        None => {
            trace!("Metrics are not initialised, skipping {key}");
        }
    }
}

/// Record a timing for a key. The time is recorded in milliseconds.
/// The function is always executed, regardless if a global measure is set or not, and the measurement is
/// written to the global measure afterwards, if present. That allows to use it in recursive settings.
pub fn r<T, F: FnOnce() -> T>(key: &str, f: F) -> T {
    r_and_accumulate::<T, F, fn(u128, u128) -> u128>(key, f, None).expect("Cannot fail")
}

/// Measure the execution time to execute the function `f`, and then record to
/// `key`` if the the measured execution time is bigger than the one found in `key`
pub fn r_if_bigger<T, F: FnOnce() -> T>(key: &str, f: F) -> anyhow::Result<T> {
    r_and_accumulate::<T, F, fn(u128, u128) -> u128>(key, f, Some(|a, b| a.max(b)))
}

/// Set a static value for a key.
pub fn set<S: ToString>(key: &str, value: S) {
    match MEASURE.lock().unwrap().as_mut() {
        Some(metrics) => {
            metrics.set(key, value);
        }
        None => {
            trace!("Metrics are not initialised, skipping {key}");
        }
    }
}

/// Allows to apply a function to the data after the measurements. The function can create new columns.
pub fn post_process<F: FnMut(&mut BTreeMap<String, String>)>(f: F) -> anyhow::Result<()> {
    match MEASURE.lock().unwrap().as_mut() {
        Some(metrics) => metrics.post_process(f),
        None => anyhow::bail!("Measures are not initialised, can't call post_process"),
    }
}

pub fn to_csv(fname: &str) -> anyhow::Result<()> {
    match MEASURE.lock().unwrap().as_ref() {
        Some(metrics) => metrics.to_csv(fname),
        None => anyhow::bail!("Measures are not initialised, can't call to_csv"),
    }
}

pub fn accumulate_key<T: ToString + FromStr<Err: Display>, F: Fn(T, T) -> T>(
    key: &str,
    value: T,
    acc_fn: F,
) -> anyhow::Result<()> {
    match MEASURE.lock().unwrap().as_mut() {
        Some(metrics) => metrics.accumulate_key(key, value, acc_fn),
        None => {
            trace!("Metrics are not initialised, skipping {key}");
            Ok(())
        }
    }
}

/// Structure that can records timing for any function calls under a key and that
/// can also hold static metrics (like "param_size"). It can be dumped on CLI or CSV
/// file at the end.
/// NOTE: this structure and all its methods are NOT thread-safe.
#[derive(Debug, Clone)]
pub struct Measure {
    data: BTreeMap<String, String>,
}

impl Measure {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }
    /// Set a static value for a key.
    pub fn with(mut self, key: &str, value: &str) -> Self {
        assert!(
            !self.data.contains_key(key),
            "Column {key} already found in predefined columns {:?}",
            self.data
        );
        self.data.insert(key.to_string(), value.to_string());
        self
    }

    /// Record a timing for a key. The time is recorded in milliseconds.
    pub fn r<T, F: FnOnce() -> T>(&mut self, key: &str, f: F) -> T {
        assert!(
            !self.data.contains_key(key),
            "Column {key} already measured {:?}",
            self.data
        );
        let (result, elapsed) = timeit(f);
        self.record_timing(key, elapsed);
        result
    }

    fn record_timing(&mut self, key: &str, elapsed: Duration) {
        self.data
            .insert(key.to_string(), elapsed.as_millis().to_string());
    }
    /// Set a static value for a key.
    pub fn set<T: ToString>(&mut self, key: &str, value: T) {
        assert!(
            !self.data.contains_key(key),
            "Column {key} already measured {:?}",
            self.data
        );
        self.data.insert(key.to_string(), value.to_string());
    }

    /// Accumulate `value `value` to the existing value of metric with key `key`, if any, using the provided
    /// closure to accumulate values
    pub fn accumulate_key<T: ToString + FromStr<Err: Display>, F: Fn(T, T) -> T>(
        &mut self,
        key: &str,
        value: T,
        acc_fn: F,
    ) -> anyhow::Result<()> {
        let current = if let Some(v) = self.data.get(key) {
            Some(T::from_str(v).map_err(|e| anyhow::anyhow!("Failed to parse {v}: {e}"))?)
        } else {
            None
        };
        let new_value = if let Some(current) = current {
            acc_fn(current, value)
        } else {
            value
        };
        self.data.insert(key.to_string(), new_value.to_string());
        Ok(())
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.data).unwrap()
    }
    /// Write the metrics to a csv file. If the file already exists, it will append to it.
    /// It will fail if the headers are not the same as the ones in the first row.
    pub fn to_csv(&self, fname: &str) -> anyhow::Result<()> {
        let file = OpenOptions::new().create(true).append(true).open(fname)?;
        // check if the file is empty, if it is, write the headers, if not, check the headers present
        // are the same
        let write_header = file.metadata().unwrap().len() == 0;
        let mut writer = csv::Writer::from_writer(file);
        if write_header {
            writer.write_record(self.data.keys().collect::<Vec<&String>>())?;
        } else {
            let mut reader = std::io::BufReader::new(File::open(fname).unwrap()).lines();
            let header_line = reader.next().unwrap()?;
            let header = header_line.split(",").collect::<Vec<&str>>();
            ensure!(header == self.data.keys().collect::<Vec<&String>>());
        }
        // iterate over all columns in order and write the values
        writer.write_record(self.data.values().map(|v| v.to_string()))?;
        writer.flush()?;
        Ok(())
    }

    /// Allows to apply a function to the data after the measurements. The function can create new columns.
    pub fn post_process<F: FnMut(&mut BTreeMap<String, String>)>(
        &mut self,
        mut f: F,
    ) -> anyhow::Result<()> {
        f(&mut self.data);
        Ok(())
    }
}

impl Display for Measure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.data
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

fn timeit<T, F: FnOnce() -> T>(f: F) -> (T, Duration) {
    let start = time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    use rand::{Rng, distr::Alphanumeric};
    use std::env;

    fn temp_filename(prefix: &str, ext: &str) -> String {
        let rand: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(12)
            .map(char::from)
            .collect();

        let mut path = env::temp_dir();
        path.push(format!("{}-{}.{}", prefix, rand, ext));
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_measure() {
        let csv_file = temp_filename("measure", "csv");
        let mut m = Measure::new().with("param_size", "1000");
        m.r("sort", || {
            let mut vec = vec![1, 2, 3, 4, 5];
            vec.sort();
            vec
        });
        m.to_csv(&csv_file).unwrap();
        assert_eq!(
            fs::read_to_string(&csv_file)
                .unwrap()
                .split("\n")
                .collect::<Vec<&str>>()[0],
            "param_size,sort"
        );
        // it should fail if we measure with a new key or out of order or different numbers
        let mut m = Measure::new().with("param_size", "1000");
        m.to_csv(&csv_file).unwrap_err();
        m.r("sort2", || {
            let mut vec = vec![1, 2, 3, 4, 5];
            vec.sort();
            vec
        });
        m.to_csv(&csv_file).unwrap_err();
    }

    #[test]
    fn test_measure_recursion() {
        let csv_file = temp_filename("measure", "csv");
        let m = Measure::new().with("param_size", "1000");
        set_global(m);
        r("sort", || {
            let mut vec = [1, 2, 3, 4, 5];
            vec.sort();
            r("sort2", || {
                let mut vec = [1, 2, 3, 4, 5];
                vec.sort();
            });
        });
        to_csv(&csv_file).unwrap();
        assert_eq!(
            fs::read_to_string(&csv_file)
                .unwrap()
                .split("\n")
                .collect::<Vec<&str>>()[0],
            "param_size,sort,sort2"
        );
    }

    #[test]
    fn test_global_measure() {
        let csv_file = temp_filename("measure", "csv");
        set_global(Measure::new().with("param_size", "1000"));
        r("sort", || {
            let mut vec = vec![1, 2, 3, 4, 5];
            vec.sort();
            vec
        });
        to_csv(&csv_file).unwrap();
        assert_eq!(
            fs::read_to_string(&csv_file)
                .unwrap()
                .split("\n")
                .collect::<Vec<&str>>()[0],
            "param_size,sort"
        );
    }
}
