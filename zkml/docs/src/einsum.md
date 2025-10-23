# EinSum Layer

EinSum is a layer for performing generalized tensor operations using Einstein summation notation. It allows you to specify complex contractions, stacking, and outer products between multiple tensors via a concise equation string, supporting both witness and constant tensors, as well as optional bias terms. This abstraction enables flexible and efficient computation for a wide range of linear algebra and neural network operations.

There are a few parts to an EinSum layer and so we have split the documentation into a number of pages to cover each of these in more detail.

- [Equation](./einsum_parts/equation.md) covers the notation used to define an EinSum layer
- [Axes Mapping](./einsum_parts/axes_mapping.md) goes into detail about how we store information about the axes of the tensors involved
- [Evaluation](./einsum_parts/evaluation.md) describes the method for evaluating the EinSum
- [Proving](./einsum_parts/proving.md) walks us through how we prove correct execution