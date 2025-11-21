//! Code for evaluating an attention mask layer

use burn::tensor::Bool;

use super::*;
use crate::backend::Backend;

impl<T> AttentionMask<T>
where
    T: TensorTypeParam,
{
    /// Apply the mask to an input, this method requires the input has rank between 2 and 4, and that the final two dims are either equal
    /// or the second to last dim is 1.
    pub(crate) fn evaluate_internal(&self, input: &WrappedTensor<T>) -> Result<WrappedTensor<T>> {
        // Reduce the input to its unpadded shape
        let current_shape = input.shape();
        let unpadded_shape = input.unpadded_shape();
        let is_padded = current_shape.as_slice() != unpadded_shape.as_slice();
        let input = if is_padded {
            input.clone().reduce_to_unpadded_shape()?
        } else {
            input.clone()
        };

        let seq_len = input.dim(-1)?;
        // // input of shape [..., num_heads, q_len, seq_len]
        // // if q_len == 1, we're in the caching inference case. Otherwise
        // // we're in the regular square matrix case where q_len == seq_len
        let caching_case = unpadded_shape.dims[unpadded_shape.num_dims() - 2] == 1
            && unpadded_shape.dims[unpadded_shape.num_dims() - 1] > 1;

        let mask = match caching_case {
            true => caching_case_mask(self.span, seq_len),
            false => non_caching_case_mask(self.span, seq_len),
        };
        let unpadded_output = input.mask_fill(mask, self.negative_infinity)?;
        if is_padded {
            Ok(unpadded_output.pad_next_power_of_two())
        } else {
            Ok(unpadded_output)
        }
    }
}

/// Makes a rank 2 mask for the caching case (so a single row).
fn caching_case_mask(span: AttentionSpan, seq_len: usize) -> BTensor<Backend, 2, Bool> {
    match span {
        AttentionSpan::Full => {
            BTensor::<Backend, 2, Bool>::full([1, seq_len], false, &Default::default())
        }
        AttentionSpan::Local(n) => {
            let offset = seq_len.saturating_sub(n) as i64;
            if offset == 0 {
                // Local span covers the whole sequence
                BTensor::<Backend, 2, Bool>::full([1, seq_len], false, &Default::default())
            } else {
                BTensor::<Backend, 2, Bool>::triu_mask([1, seq_len], offset, &Default::default())
            }
        }
    }
}

/// Makes a rank 2 mask for the non-caching case (so a square matrix).
fn non_caching_case_mask(span: AttentionSpan, seq_len: usize) -> BTensor<Backend, 2, Bool> {
    let plain_mask =
        BTensor::<Backend, 2, Bool>::tril_mask([seq_len, seq_len], 0, &Default::default());
    match span {
        AttentionSpan::Full => plain_mask,
        AttentionSpan::Local(n) => {
            let offset = (seq_len - 1).saturating_sub(n) as i64 - 1;
            // If the offset is negative, it means the local span covers the whole matrix so we just return the plain mask
            if offset < 0 {
                plain_mask
            } else {
                let extra_mask = BTensor::<Backend, 2, Bool>::triu_mask(
                    [seq_len, seq_len],
                    -offset,
                    &Default::default(),
                );
                plain_mask.bool_or(extra_mask)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use burn::tensor::Shape as BShape;

    #[derive(Debug, Clone, Copy)]
    struct MaskTestCase {
        seq_len: usize,
        q_len: usize,
        num_heads: usize,
        span: AttentionSpan,
    }

    #[test]
    fn test_mask_evaluation() -> Result<()> {
        let test_cases = vec![
            MaskTestCase {
                seq_len: 3,
                q_len: 3, // regular square matrix case
                num_heads: 2,
                span: AttentionSpan::Local(2),
            },
            MaskTestCase {
                seq_len: 3,
                q_len: 1, // caching inference
                num_heads: 2,
                span: AttentionSpan::Local(2),
            },
            MaskTestCase {
                seq_len: 3,
                q_len: 1, // caching inference
                num_heads: 2,
                span: AttentionSpan::Full,
            },
            MaskTestCase {
                seq_len: 3,
                q_len: 3, // regular square matrix case
                num_heads: 2,
                span: AttentionSpan::Full,
            },
        ];
        for test_case in test_cases {
            test_mask_evaluation_helper::<f32>(test_case)?;
            test_mask_evaluation_helper::<Element>(test_case)?;
        }
        Ok(())
    }

    fn test_mask_evaluation_helper<T: TensorTypeParam>(test_case: MaskTestCase) -> Result<()> {
        let MaskTestCase {
            seq_len,
            q_len,
            num_heads,
            span,
        } = test_case;

        let shape: Shape = vec![num_heads, q_len, seq_len].into();
        let bshape: BShape = shape.clone().into();
        let expected_mask = if q_len == 1 {
            caching_case_mask(span, seq_len).expand::<3, _>(bshape)
        } else {
            non_caching_case_mask(span, seq_len).expand::<3, _>(bshape)
        };

        let input = WrappedTensor::<T>::random(&shape);
        let attention_mask =
            AttentionMask::<T>::new(seq_len, <T as Number>::MIN).with_span(span)?;

        let output = attention_mask.evaluate_internal(&input)?;

        // Check the output shape matches the input shape
        assert_eq!(output.shape().as_slice(), shape.as_slice());

        // Check the mask was applied correctly
        let neg_inf = <T as Number>::MIN;
        let masked_portion = output.equal_elem(neg_inf)?;
        let is_equal = masked_portion
            .equal(expected_mask)
            .into_data()
            .iter::<bool>()
            .all(|b| b);
        assert!(
            is_equal,
            "Mask not applied correctly for test case: {:?}",
            test_case
        );
        Ok(())
    }
}
