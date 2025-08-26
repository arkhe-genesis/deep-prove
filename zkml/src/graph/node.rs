/// A trait for operations that can be executed on a graph.
/// It is used to define the input and output types of the operation.
/// It is also used to define the run method that will be used to execute the operation.
pub trait GraphNode: Clone {
    /// The input and output type for the node
    type IO: Clone;
    /// The context necessary for the node to execute the operation. This method is meant to be
    /// called either locally or on remote worker. The context
    /// can hold references to the setup parameters that we don't want to send over the wire.
    type Context;
    /// A description of the node, helpful for debugging and logging purposes.
    fn describe(&self) -> String;
    /// Runs the operation with the given context and inputs.
    /// The inputs comes from the graph processing (output of predecessor nodes).
    fn run(&self, ctx: &Self::Context, inputs: Vec<Self::IO>) -> anyhow::Result<Self::IO>;
}

/// Helper macro to extract a variant from a vector of enums.
/// This is useful for nodes of the graph which are variants, so when one needs to extract
/// the inputs from a vector of enums, it can do so via this macro.
#[allow(unused_macros)]
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
