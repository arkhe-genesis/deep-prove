# RMSNorm

# Description of the Layer

RMSNorm normalises a tensor along a certain dimension by dividing by the "root mean squared". That is 

$$ \begin{align*} \mathrm{RMSNorm}(A)_{i,j} := \alpha_{j} \cdot \frac{A_{i,j}}{\sqrt{\frac{1}{n}\cdot\sum_{l=1}^{n}A_{i,l}^{2} + \epsilon}}.\end{align*} $$

Here we have that $`\alpha_{j}`$ is a learned constant and $`\epsilon`$ is a normalisation factor.

## Precision

When we calculate the term $`n\cdot \hat{\nu}_{i} - \hat{\mu}_{i}^{2}`$ to pass to the lookup we don't actually use the full value as the range of possible values it could take is far too large. Instead we only use the most significant $`2b_{Q}`$ bits, where $`b_{Q}`$ is the quantisation bit-length.

To see why we do this consider the following example. We pick two numbers $`3.4372`$ and $`4.5133`$ and their product is $`15.51311476`$. If we round both to three significant figures then we have $`3.44`$ and $`4.51`$, their product is $`15.5144`$ which we can see differs at the 5th significant figure, so using anything beyond this isn't going to give any more accuracy in the final result (as it is already wrong).

## Quantised Evaluation

The main difficulty comes from computing the inverse square root term. For this we use a lookup table that takes as input $`\sum_{l=1}^{n}A_{i,l}^{2}`$ and outputs $`D_{i} = (1/n \cdot \sum_{l=1}^{n}A_{i,l}^{2} + \epsilon)^{-1/2}`$. The final layer output is then calculated by performing the multiplication $`\alpha_{j}\cdot A_{i,j}\cdot D_{i}`$. 

## Proving the Layer

To prove the correct execution of RMSNorm we use a combination of lookups and standard sumchecks. The lookup protocol is used to prove correct computation of 

$$ \begin{align*} D_{i} := \frac{1}{\sqrt{\frac{1}{n} \cdot \sum_{l=1}^{n}A_{i,l}^{2} + \epsilon}} \end{align*} $$

and then a standard product sumcheck is used to prove that $`\alpha_{j} \cdot A_{i,j}* D_{i} = \mathrm{RMSNorm}(A)_{i,j}
`$ element-wise.

### Step-by-Step

The prover receives the input tensor $`A`$ and its corresponding MLE $`A(\bar{x})`$. They use this to compute the input to the lookup table $`\mathrm{LookupIn}`$ and output of the lookup table $`D`$ together with their corresponding MLEs $`\mathrm{LookupIn}(\bar{x})`$ and $`D(\bar{x})`$. 

To compute $`\mathrm{LookupIn}`$ the prover calculates $`\sum_{l=1}^{n} A_{i,l}^{2} `$. The outputs of this sum have large bit size, too large for a single lookup table, so if the quantisation bit length is $`b_{q}`$ we use the fact that only the most significant $`2b_{q}`$ bits of the sum have any real precision (any bits less significant than this are calculated via terms that involve rounding error). We split the most significant $`2b_{q}`$ (see [this](./layernorm.md#precision) for more info) bits of the sum off to become $`\mathrm{LookupIn}`$ and range check the remainder.

 Now the prover commits to both $`\mathrm{LookupIn}(\bar{x})`$, $`D(\bar{x})`$ and the range check lookup, appending the commitments to the transcript.

They run the lookup argument to obtain claims $`\mathrm{LookupIn}(\bar{s_{1}'}) = u`$, $`D(\bar{s_{1}'})=w`$ and $`\mathrm{Range}(\bar{s_{1}'}) = t`$.

They check via sumcheck that

$$ \begin{align} 2^{\mathrm{rangebits}}\cdot\mathrm{LookupIn}(\bar{s}) + \mathrm{Range}(\bar{s}) =& \sum_{b\in\mathcal{B}_{m}} \mathrm{eq}(2^{-1},\dots,2^{-1},s, b)(2^{\lceil\log{n}\rceil} A(b)^{2})\end{align} $$


The prover also has the claim about the RMSNorm output which is a point $`\bar{r}`$ and a value $`v`$. They use this to reduce the claim on the output to a claim on $`A(\bar{x})`$ and $`D(\bar{x})`$ by running the sumcheck:

$$ \begin{align} v=\sum_{b\in\mathcal{B}_{m}}\mathrm{eq}(\bar{r},b)\cdot \alpha(b)\cdot \hat{D}(b)\cdot A(b)\end{align} $$

Here $`\hat{D}(\bar{x})`$ is an extension of $`D(\bar{y})`$ such that $`D(\bar{y}) = \hat{D}(a_{1},\cdots,a_{\lceil\log{n}\rceil,\bar{r}})`$ for any choice of the $`a_{i}`$.

These two checks are batched together, the claim the prover creates on $`\hat{D}`$ is verified by commitment opening.

This final sumcheck provides use with the claim that is passed to the next layer.