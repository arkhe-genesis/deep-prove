# EinSum Evaluation

To evaluate the EinSum we use the fact that any valid equation (as defined in [Axes Mapping](./axes_mapping.md)) can be transformed via linear maps into a product of 3D tensors.

Given a valid einsum equation, each operation can be represented as a contraction over a subset of axes, stacking over shared axes, and outer product over unique axes. For tensors $A$ and $B$ with axes partitioned as follows:

- $\mathbf{s}$: stacking axes (present in both inputs and output)
- $\mathbf{c}$: contraction axes (present in both inputs, not in output)
- $\mathbf{o}_A$: outer axes unique to $A$ and output
- $\mathbf{o}_B$: outer axes unique to $B$ and output

The einsum operation can be written as:

$$
C_{\mathbf{s}, \mathbf{o}_A, \mathbf{o}_B} = \sum_{\mathbf{c}} A_{\mathbf{s}, \mathbf{c}, \mathbf{o}_A} \cdot B_{\mathbf{s}, \mathbf{c}, \mathbf{o}_B}
$$

where the sum is taken over all contraction axes $\mathbf{c}$, and the output tensor $C$ is indexed by the stacking and outer axes.

Any valid einsum with more than two tensors can be decomposed into a sequence of such 3D contractions by grouping axes and tensors appropriately, so that each step is a contraction over a set of axes, with the remaining axes corresponding to stacking and outer axes.

This transformation allows the computation to be performed as a series of 3D tensor multiplications and summations, making the operation efficient and generalizable.

## Example

Suppose we have tensors $`A`$ and $`B`$, $`A`$ has axes $`(i, j, k, l)`$ and $`B`$ has axes $`(k, l)`$. They have output $`C`$ with axes $`(i, j)`$. Then to view this as a 3D operation we note that in this case $`\mathbf{s} = 1`$, $`\mathbf{c} = k\times l`$, $`\mathbf{o}_{A} = i \times j`$ and $`\mathbf{o}_{B} = 1`$. So we can view $`A \in \mathbb{R}^{1\times(i \times j)\times(k\times l)}`$ and $`B \in \mathbb{R}^{1\times(k \times l)\times 1}`$. Then the action of $`A`$ on $`B`$ is a batched matrix multiplication (here the total number of batches is 1, in general is equal to the product of all the stacking axes sizes). Then the output of this batched matrix multiplication is the intermediate tensor $`\mathrm{Int} \in \mathbb{R}^{1\times(i \times j)\times 1}`$ and then we would map $`\mathrm{Int}`$ to $`C`$ by reshaping.