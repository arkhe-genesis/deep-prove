use candle_core::quantized::{QTensor, gguf_file::Value};
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Context, bail, ensure};
use candle_core::{CpuStorage, Device, Storage, quantized::gguf_file::Content};

use crate::{
    Shape, Tensor,
    tensor::{KeyedTensor, TensorKey},
};

fn dequantize(qtensor: &QTensor) -> anyhow::Result<Tensor<f32>> {
    let shape = Shape::new(qtensor.shape().dims().to_vec());

    let dequantized_candle_tensor = qtensor
        .dequantize(&Device::Cpu)
        .map_err(anyhow::Error::from) // Convert candle_core::Error to anyhow::Error
        .with_context(|| {
            format!(
                "Failed to dequantize QTensor (dtype: {:?}, shape: {:?})",
                qtensor.dtype(),
                qtensor.shape()
            )
        })?;

    let (s, _l) = dequantized_candle_tensor.storage_and_layout();
    let data: Vec<f32> = match s.deref() {
        Storage::Cpu(cpu_storage) => match cpu_storage {
            CpuStorage::F32(d) => d.to_vec(),
            CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
            CpuStorage::BF16(d) => d.iter().map(|x| x.to_f32()).collect(),
            _ => bail!(
                "Dequantization resulted in an unexpected quantized CPU storage type (original QTensor dtype: {:?})",
                qtensor.dtype()
            ),
        },
        // Change storage_device() to device()
        _ => bail!(
            "Unsupported storage backend for dequantized tensor (expected CPU), got: {:?}",
            dequantized_candle_tensor.device()
        ),
    };
    Ok(Tensor::new(shape, data))
}

pub fn unfuse_tensors(
    fused: candle_core::Tensor,
    chunk_len: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let (s, _l) = fused.storage_and_layout();
    let data: Vec<f32> = match s.deref() {
        Storage::Cpu(cpu) => match cpu {
            CpuStorage::F32(d) => d.to_vec(),
            CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
            _ => bail!(
                "unsupported storage type (only f32 or f16 is supported for unfusing candle::Tensor)"
            ),
        },
        _ => {
            bail!("unsupported storage backend (only cpu is supported for unfusing candle::Tensor)")
        }
    };
    let num_elements = data.len();
    ensure!(
        num_elements.is_multiple_of(chunk_len),
        "Total elements {num_elements} is not divisible by chunk_len {chunk_len} for unfusing"
    );
    let tensors: Vec<Vec<f32>> = data
        .chunks_exact(chunk_len)
        .map(|chunk| chunk.to_vec())
        .collect();
    Ok(tensors)
}

pub trait FromValue<T> {
    fn from_value(v: &Value) -> T;
}

impl FromValue<f32> for Value {
    fn from_value(v: &Value) -> f32 {
        v.to_f32().expect("failed to convert f32 to f32")
    }
}

impl FromValue<f64> for Value {
    fn from_value(v: &Value) -> f64 {
        v.to_f64().expect("failed to convert f64 to f64")
    }
}
impl FromValue<usize> for Value {
    fn from_value(v: &Value) -> usize {
        v.to_u32().expect("failed to convert u32 to u32") as usize
    }
}

impl FromValue<Vec<Value>> for Value {
    fn from_value(v: &Value) -> Vec<Value> {
        v.to_vec()
            .expect("failed to convert Value to Vec<Value>")
            .to_vec()
    }
}

impl FromValue<String> for Value {
    fn from_value(v: &Value) -> String {
        v.to_string()
            .expect("failed to convert Value to String")
            .clone()
    }
}

impl FromValue<u32> for Value {
    fn from_value(v: &Value) -> u32 {
        v.to_u32().expect("failed to convert Value to u32")
    }
}

/// Type alias for a TensorLoader specialized for reading from a BufReader<File>.
/// This simplifies the instantiation when loading tensors directly from a file path.
pub type FileTensorLoader = TensorLoader<BufReader<File>>;

#[derive(Clone)]
/// Manages lazy loading of tensors from a GGUF file.
///
/// This structure allows for efficient, on-demand loading of tensor data.
/// It supports sub-scoping for tensor names (e.g., `blk.0.attn_norm.weight`)
/// by maintaining an internal prefix. It is designed to be cloneable, making
/// it easy to pass around or use in different parts of a model definition.
/// Tensor loading is deferred until a specific tensor is requested via `get_tensor`.
pub struct TensorLoader<R: Read + Seek> {
    /// Parsed GGUF metadata and tensor information.
    /// This is an `Arc` to allow cheap cloning of the `TensorLoader`.
    content: Arc<Content>,
    /// Reader for the GGUF file, allowing lazy loading of tensor data.
    /// It's wrapped in `Arc<Mutex<>>` to enable shared, mutable access
    /// across cloned instances and for thread-safety if used in concurrent contexts.
    reader: Arc<Mutex<R>>,
    /// Current prefix for tensor names. When a tensor is requested,
    /// this prefix is prepended to the requested name to form the full tensor name.
    current_prefix: String,
    /// The `Device` on which `QTensor`s (quantized tensors) should be initially loaded.
    /// Note: The existing `dequantize` function subsequently converts these to `crate::Tensor<f32>`
    /// and currently materializes them on the CPU.
    device: Device,
}

impl<R: Read + Seek + Send + 'static> TensorLoader<R> {
    /// Creates a new `TensorLoader` from a given reader and device.
    /// The reader must be positioned at the beginning of the GGUF file.
    ///
    /// # Arguments
    /// * `reader` - A type implementing `Read` and `Seek` for the GGUF file (e.g., `BufReader<File>`).
    ///
    /// # Errors
    /// Returns an error if reading the GGUF content metadata fails.
    pub fn from_reader(mut reader: R) -> anyhow::Result<Self> {
        let content = Content::read(&mut reader)?;
        Ok(Self {
            content: Arc::new(content),
            reader: Arc::new(Mutex::new(reader)),
            current_prefix: String::new(),
            device: Device::Cpu,
        })
    }

    /// Creates a new `TensorLoader` instance representing a sub-scope.
    /// The new scope's prefix is formed by concatenating the current loader's prefix
    /// with the `prefix_extension`. For example, if the current prefix is `blk.0.`
    /// and `prefix_extension` is `attn_`, the new prefix will be `blk.0.attn_`.
    ///
    /// # Arguments
    /// * `prefix_extension` - The string to append to the current prefix to define the new scope.
    ///
    /// # Returns
    /// A new `TensorLoader` instance for the specified sub-scope.
    pub fn pp(&self, prefix_extension: &str) -> Self {
        Self {
            content: Arc::clone(&self.content),
            reader: Arc::clone(&self.reader),
            current_prefix: format!("{}{}", self.current_prefix, prefix_extension),
            device: self.device.clone(),
        }
    }

    pub fn get_tensor_shape(&self, name: &str) -> anyhow::Result<Shape> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))?;
        Ok(Shape::new(info.shape.dims().to_vec()))
    }

    /// Retrieves a quantized tensor (`QTensor`) by its name relative to the current scope.
    /// The full tensor name is formed by `current_prefix + name`.
    /// This method is primarily for internal use or advanced scenarios where the `QTensor` is needed directly.
    ///
    /// # Arguments
    /// * `name` - The name of the tensor, relative to the current scope (e.g., `weight`).
    ///
    /// # Errors
    /// Returns an error if the reader lock cannot be acquired or if `Content::tensor` fails to load the `QTensor`.
    pub(crate) fn get_qtensor(&self, name: &str) -> anyhow::Result<(TensorKey, Arc<QTensor>)> {
        let full_name = format!("{}{}", self.current_prefix, name);
        let mut reader_guard = self.reader.lock().map_err(|e| {
            anyhow::anyhow!("Failed to acquire reader lock for tensor '{full_name}': {e}")
        })?;
        self.content
            .tensor(&mut *reader_guard, &full_name, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to load QTensor '{full_name}' from GGUF: {e}"))
            .map(|qtensor| (full_name.into(), Arc::new(qtensor)))
    }

    /// Retrieves and dequantizes a tensor by its name relative to the current scope.
    ///
    /// This method first loads the quantized tensor (`QTensor`) using `get_qtensor`,
    /// then calls the `dequantize` function (expected to be available in the same module)
    /// to convert it into a `crate::Tensor<f32>`.
    ///
    /// # Arguments
    /// * `name` - The name of the tensor, relative to the current scope (e.g., `attn_norm.weight`).
    ///
    /// # Errors
    /// Returns an error if `get_qtensor` fails or if the subsequent dequantization fails.
    pub fn get_tensor(&self, name: &str) -> anyhow::Result<KeyedTensor<f32>> {
        let (key, qtensor) = self.get_qtensor(name)?;
        let tensor = dequantize(qtensor.as_ref())?;
        Ok(KeyedTensor::new(key, tensor))
    }

    pub fn metadata<T>(&self, key: &str) -> Option<T>
    where
        Value: FromValue<T>,
    {
        self.content.metadata.get(key).map(Value::from_value)
    }

    pub fn raw_metadata(&self, key: &str) -> Option<&Value> {
        self.content.metadata.get(key)
    }
}

impl TensorLoader<BufReader<File>> {
    /// Creates a new `TensorLoader` by opening and reading a GGUF file from the specified path.
    ///
    /// # Arguments
    /// * `path` - The file system path to the GGUF file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or if reading the GGUF content metadata fails.
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to open file {:?}: {}", path.as_ref(), e))?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    pub fn meta_and_tensor_keys(&self) -> (Vec<String>, Vec<String>) {
        let mut meta_keys = self.content.metadata.keys().cloned().collect::<Vec<_>>();
        let mut tensor_keys = self
            .content
            .tensor_infos
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        meta_keys.sort();
        tensor_keys.sort();
        (meta_keys, tensor_keys)
    }
}

#[cfg(test)]
pub mod tests {
    use candle_core::{
        CpuStorage, Device, Storage,
        quantized::gguf_file::{Content, Value},
    };
    use gguf_rs::get_gguf_container;
    use std::{fs::File, ops::Deref};

    use crate::{
        layers::transformer::embeddings::Embeddings,
        parser::{
            file_cache,
            llm::{HFTokenizer, LLMConfig, LLMTokenizer, LLMVariant, transformer::Attention},
        },
    };

    // download at https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF
    // pub const GPT2_Q8_0_PATH: &str = "assets/scripts/llms/gpt2.q8_0.gguf";
    // const GPT2_Q8_0_URL: &str = "https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.q8_0.gguf";
    pub const GPT2_Q8_0: &str = "gpt2.Q8_0.gguf";
    pub const GEMMA3_Q8: &str = "gemma-3-270m-it-Q8_0.gguf";

    #[test]
    fn test_gguf_load_model() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let config = LLMConfig::from_content(&loader)?;
        let _model = config.model(&loader)?;
        println!("model: {:?}", config.variant);
        Ok(())
    }

    #[test]
    fn test_gguf_load_attention() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let config = LLMConfig::from_content(&loader)?;
        let block0_loader = loader.pp("blk.0.");

        let _attention = Attention::from_loader(&block0_loader, &config)?;
        Ok(())
    }

    #[test]
    fn test_gguf_load_config() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let config = LLMConfig::from_content(&loader)?;
        println!("config: {config:?}");
        Ok(())
    }

    #[test]
    fn test_gguf_load_embedding() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let _embedding = Embeddings::from_loader(&loader)?;
        Ok(())
    }

    // https://docs.rs/candle-transformers/latest/src/candle_transformers/models/llama.rs.html#517-535
    #[test]
    //#[ignore = "just a test to explore gguf internal structure"]
    fn test_load_and_inspect_gpt2_gguf() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;

        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Model path is not valid UTF-8"))?;
        let mut container = get_gguf_container(model_path_str)?;
        let model = container.decode()?;

        println!("GGUF version: {}", model.get_version());
        println!("GGUF metadata: {:?}", model.metadata());
        let mut r = File::open(model_path)?;
        let gguf_candle = Content::read(&mut r)?;
        println!("GGUF metadata: {:?}", gguf_candle.metadata.keys());
        // println!("token length: {:?}", gguf_candle.metadata.get("tokenizer.ggml.tokens"));
        // println!("token merges: {:?}", gguf_candle.metadata.get("tokenizer.ggml.merges"));
        println!(
            "token special: {:?}",
            gguf_candle.metadata.get("tokenizer.ggml.special_tokens")
        );
        // println!("GGUF tensors: {:?}", gguf_candle.tensor_infos);
        // println!("GGUF tensors: {:?}", model.tensors().iter().map(|t| t.name.clone()).collect::<Vec<_>>());
        for tensor in model.tensors() {
            // println!("Tensor name: {}", tensor.name);
            // println!("Tensor kind: {}", tensor.kind);
            let _num_elements = tensor.shape.iter().product::<u64>();
            // println!(
            //    "Tensor shape: {:?} -> total {:?}",
            //    tensor.shape, num_elements
            //);
            let qtensor = gguf_candle.tensor(&mut r, &tensor.name, &Device::Cpu)?;
            let tensor = qtensor.dequantize(&Device::Cpu)?;
            let (s, _l) = tensor.storage_and_layout();
            let _data = match s.deref() {
                Storage::Cpu(s) => match s {
                    CpuStorage::F32(d) => d.to_vec(),
                    CpuStorage::F16(d) => d.iter().map(|x| x.to_f32()).collect(),
                    _ => {
                        panic!("unsupported type of tensor: {s:?}");
                    }
                },
                _ => {
                    panic!("only cpu storage type is supported");
                }
            };
        }
        Ok(())
    }

    use crate::parser::gguf::FileTensorLoader;
    #[test]
    fn test_tensor_loader_subscoping_and_lazy_load() -> anyhow::Result<()> {
        // let gguf_path = GPT2_Q8_0_PATH;
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;

        // Create TensorLoader using the type alias
        let loader = FileTensorLoader::from_path(model_path)?;

        // Test loading a tensor from the root scope
        let embedding_tensor = loader.get_tensor("token_embd.weight")?.into_tensor();
        // Expected shape for gpt2 token_embd.weight: [vocab_size, embedding_length] = [50257, 768]
        assert_eq!(
            *embedding_tensor.shape(),
            vec![50257usize, 768usize].into(),
            "Shape mismatch for token_embd.weight"
        );

        // Test sub-scoping with a trailing dot (VarBuilder style)
        let blk0_loader = loader.pp("blk.0.");
        let attn_norm_weight = blk0_loader.get_tensor("attn_norm.weight")?.into_tensor();
        // Expected shape for blk.0.attn_norm.weight: [embedding_length] = [768]
        assert_eq!(
            *attn_norm_weight.shape(),
            vec![768usize].into(),
            "Shape mismatch for blk.0.attn_norm.weight"
        );

        let qkv_weight = blk0_loader.get_tensor("attn_qkv.weight")?.into_tensor();
        // Expected shape for blk.0.attn_qkv.weight: [3 * embedding_length, embedding_length] = [2304, 768]
        assert_eq!(
            *qkv_weight.shape(),
            vec![2304usize, 768usize].into(),
            "Shape mismatch for blk.0.attn_qkv.weight"
        );

        // Test sub-scoping with custom prefix as requested ("attn_", "ffn_")
        // Current prefix of blk0_loader is "blk.0."
        let blk0_attn_loader = blk0_loader.pp("attn_"); // New prefix: "blk.0.attn_"
        let attn_norm_weight_v2 = blk0_attn_loader.get_tensor("norm.weight")?.into_tensor(); // Full name: "blk.0.attn_norm.weight"
        assert_eq!(
            *attn_norm_weight_v2.shape(),
            vec![768usize].into(),
            "Shape mismatch for blk.0.attn_norm.weight via custom subscope"
        );

        let blk0_ffn_loader = blk0_loader.pp("ffn_"); // New prefix: "blk.0.ffn_"
        let ffn_norm_weight = blk0_ffn_loader.get_tensor("norm.weight")?.into_tensor(); // Full name: "blk.0.ffn_norm.weight"
        // Expected shape for blk.0.ffn_norm.weight: [embedding_length] = [768]
        assert_eq!(
            *ffn_norm_weight.shape(),
            vec![768usize].into(),
            "Shape mismatch for blk.0.ffn_norm.weight via custom subscope"
        );

        // Test that loading a non-existent tensor fails
        let non_existent_tensor_result = blk0_loader.get_tensor("non_existent_tensor.weight");
        assert!(
            non_existent_tensor_result.is_err(),
            "Expected error for non-existent tensor"
        );

        Ok(())
    }

    #[test]
    fn test_gguf_load_tokenizer() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let tokenizer = HFTokenizer::from_loader(&loader)?;
        let s = "do or don't. there is no try.";
        let tokens = tokenizer.tokenize(s);
        let s2 = tokenizer.detokenize(&tokens);
        assert_eq!(s, s2);
        Ok(())
    }

    #[test]
    fn test_gguf_print_keys() -> anyhow::Result<()> {
        for path in [GPT2_Q8_0, GEMMA3_Q8] {
            let model_path = file_cache::from_cache(path)?;
            let loader = FileTensorLoader::from_path(model_path)?;
            let (meta_keys, tensor_keys) = loader.meta_and_tensor_keys();
            println!("{path}");
            println!("metadata:");
            for key in meta_keys {
                match loader.raw_metadata(&key).unwrap() {
                    Value::String(s) => println!(
                        "\t- {key} (string): {}",
                        &s.chars().take(10).collect::<String>()
                    ),
                    Value::F32(f) => println!("\t- {key} (f32): {f}"),
                    Value::F64(f) => println!("\t- {key} (f64): {f}"),
                    Value::Bool(b) => println!("\t- {key} (bool): {b}"),
                    Value::I8(i) => println!("\t- {key} (i8): {i}"),
                    Value::I16(i) => println!("\t- {key} (i16): {i}"),
                    Value::I32(i) => println!("\t- {key} (i32): {i}"),
                    Value::I64(i) => println!("\t- {key} (i64): {i}"),
                    Value::U8(u) => println!("\t- {key} (u8): {u}"),
                    Value::U16(u) => println!("\t- {key} (u16): {u}"),
                    Value::U32(u) => println!("\t- {key} (u32): {u}"),
                    Value::U64(u) => println!("\t- {key} (u64): {u}"),
                    Value::Array(v) => println!("\t- {key}: {:?}", v.len()),
                }
            }
            println!("tensor_infos:");
            for key in tensor_keys {
                println!("\t- {key}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_gguf_gemma3() -> anyhow::Result<()> {
        let gemma = GEMMA3_Q8;
        let model_path = file_cache::from_cache(gemma)?;
        let loader = FileTensorLoader::from_path(model_path)?;
        let (meta_keys, tensor_keys) = loader.meta_and_tensor_keys();
        println!("metadata:");
        for key in meta_keys {
            println!("\t- {key}");
        }

        println!("tensor_infos:");
        for key in tensor_keys {
            println!("\t- {key}");
        }
        println!(" VARIANT: {:?}", LLMVariant::from_loader(&loader)?);

        Ok(())
    }
}
