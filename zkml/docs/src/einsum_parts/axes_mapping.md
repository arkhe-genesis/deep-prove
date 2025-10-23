# Axes Mapping

The axes mapping stores all the information about where each axis appears within the EinSum so it can then be used to do things like:

- Verify input shapes
- Compute output shapes
- Fix variables corresponding to certain axes in tensor MLEs


## Equation 

A valid EinSum equation must follow these rules:

- Each tensor is identified by an uppercase name followed by its axes in parentheses, e.g. `A(ijk)`.
- The leftmost tensor (the "LHS") is always a witness tensor and cannot be a constant.
- The LHS acts individually on each other input tensor, separated by `@` and `:`, e.g. `A(ijk)@B(ikl):C(iml)`.
- Axes are represented by lowercase letters. Each axis label must be unique within a tensor.
- The output tensors are specified after `->`, with their axes in parentheses, e.g. `->X(ijl):Y(ikl)`.
- `Stacking` axes appear in both inputs and outputs.
- `Contraction` axes appear in both inputs but not in the output, and must be present in the same order in all input tensors. There must be at least one contraction axis.
- `Outer` axes appear in exactly one input and the output.
- The number of output tensors must match the number of einsum operations.
- Bias tensors, if present, are indicated by `+BIAS(...)` after the output. The bias axes should always be a subset of the axes of the output it is added to.

These rules ensure the equation is well-formed and unambiguous for parsing and computation.

They are slightly different to the rules found in some other libraries. This is because we aim to improve proving efficiency where ever possible and so if one tensor acts on many different tensors (for instance when generating `Q`, `K` and `V` in attention) we wish to batch this into a single layer. In the `QKV` case mention the equation would look like: 

```
X(se)@WQ(eh):WK(eh):WV(eh)->Q(sh)+BIAS(h):K(sh)+BIAS(h):V(sh)+BIAS(h)
```

Finally the axes mapping is "agnostic" to any concrete shape, inputs are valid as long as all of their dimension sizes agree on like axes. In addition it also handles permuting inputs implicitly, so one does not have to permute tensors before passing them to the EinSum.

## `AxesMapping` Struct

The `struct` we use to store all this information once it has been parsed is:

```
pub struct AxesMapping {
    /// The number of input tensors in the operation.
    input_count: usize,
    /// The number of output tensors in the operation.
    output_count: usize,
    /// The number of bias tensors in the operation.
    bias_count: usize,
    /// A list of all axes involved in the operation.
    axes: Vec<Axis>,
}
```

Where `Axis` is defined as:

```
pub struct Axis {
    /// This indicates whether the LHS of the operation has this axs present
    pub(crate) lhs_input: Dimension,
    /// This vector indicates the presence or absence of the axis in each input tensor on the RHS of the operation.
    pub(crate) rhs_inputs: Vec<Dimension>,
    /// This vector indicates the presence or absence of the axis in each output tensor of the operation.
    pub(crate) outputs: Vec<Dimension>,
    /// This vector indicates the presence or absence of the axis in each bias tensor of the operation.
    pub(crate) biases: Vec<Dimension>,
    /// A character representation of the axis, used in Einstein summation notation.
    pub repr: char,
    /// The type of the axis in the EinSum operation.
    pub axis_type: AxisType,
}
```
The `Dimension` `enum` has two variants `Absent` indicating that the specified tensor does not use this `Axis`, or `Present(usize)`. The inner `usize` in the `Present` variant indicates the index of this `Axis` in the tensors `Shape`. For example if we had a tensor `A(ijk)` the `Axis` with `repr == i` would store `Dimension::Present(0usize)` for this tensor, similarly the `Axis` with `repr == k` would store `Dimension::Present(2usize)`. However an `Axis` with `repr == l` would store `Dimension::Absent`.

Finally `AxisType` is an `enum` that tells us whether in the mapping this `Axis` is `Stacking`, `Outer` or `Contracted`.

## Fixing Axes in MLEs

When proving an EinSum the MLEs used to represent input tensors must have variables corresponding to their `Outer` axes fixed and parts must be rescaled corresponding to their `Stacking` axes.

To illustrate this we give an example. Say we have an EinSum with equation `A(ijk)@B(ikl)->C(ijl)`, then `i` is a `Stacking` axis, `j` and `l` are `Outer` axes and `k` is the `Contraction` axis. For concrete numbers let `i = j = l = 2` and `k = 3`. Then we can have 
```
A = [
    [
        [a1, a2, a3],
        [a4, a5, a6]
    ],
    [
        [a7, a8, a9],
        [a10, a11, a12]
    ]
]
```
and
```
B = [
    [
        [b1, b2],
        [b3, b4],
        [b5, b6]
    ],
    [
        [b7, b8],
        [b9, b10],
        [b11, b12]
    ]
]
```
If the evaluation claim point on `C` is `r = (rl, rj, ri)` then the fixing occurs as follows. First we fix the `Outer` axis of `A` using `rj` resulting in:
```
A_rj = [
    [
        (1 - rj)a1 + rja4, (1 - rj)a2 + rja5, (1 - rj)a3 + rja6
    ],
    [
        (1 - rj)a7 + rja10, (1 - rj)a8 + rja11, (1 - rj)a9 + rja12
    ],
]
```

Then the two separate parts would be split along the `Stacking` axis `i` and have coefficients calculated from `ri` as 
```
A_rj_1 = (1 - ri) * [
        (1 - rj)a1 + rja4, 
        (1 - rj)a2 + rja5, 
        (1 - rj)a3 + rja6
    ]

A_rj_2 = ri * [
        (1 - rj)a1 + rja4, 
        (1 - rj)a2 + rja5, 
        (1 - rj)a3 + rja6
    ]
```
This gives us two MLEs for `A`, one for each of the "heads" in the `Stacking` dimension `i`.

Similarly for `B` we fix the `Outer` axis at `rl`
```
B_rl = [
    [(1-rl)b1 + rlb2, (1-rl)b3 + rlb4, (1-rl)b5 + rlb6],
    [(1-rl)b7 + rlb7, (1-rl)b9 + rlb10, (1-rl)b11 + rlb12],
]
```
Then we split this along the `Stacking` axis `i` as well obtaining 
```
B_rl_1 = (1 - ri). * [
    (1-rl)b1 + rlb2, 
    (1-rl)b3 + rlb4, 
    (1-rl)b5 + rlb6
]

B_rl_2 = ri * [
    (1-rl)b7 + rlb7, 
    (1-rl)b9 + rlb10, 
    (1-rl)b11 + rlb12
]
```

An extra `0` would then be appended to each of the above evaluation lists in order to retrieve a power of two number of evaluations.