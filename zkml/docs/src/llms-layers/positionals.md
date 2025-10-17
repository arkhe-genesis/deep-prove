# Positional Encoding
Positional encoding layer in LLMs, given as input an embedding vector associated to a token, encodes in the vector some data about the position of the token in the full sequence of tokens processed by the LLM.
There are 2 types of positional encoding in LLMs supported in DeepProve: **absolute** positional encoding and **rotary** positional encoding (labelled as RoPE). We describe here how to prove both variants of positional encoding.

## Absolute Positional Encoding
The data about each position of a token is encoded in a vector with $\mathtt{embedding\_size}$ elements learnt during the training of the model, called *positional encoding vector*. There is a distinct positional encoding vector $p_i \in \mathbb{R}^{\mathtt{embedding\_size}}$, for each $i \in \{0, \dots, \mathtt{context\_length}-1\}$, where $\mathtt{context\_length}$ is an hyper-parameter defining the maximum number of tokens that can be processed by the LLM model. All the positional encoding vectors are collected in a matrix $P \in \mathbb{R}^{\mathtt{context\_length} \times \mathtt{embedding\_size}} = [p_0, \dots, p_{\mathtt{context\_length}-1}]$, that is the i-th row of $P$ corresponds to the positional encoding vector $p_i$. In the following, for the sake of brevity, we will denote the $\mathtt{embedding\_size}$ with $e$ and $\mathtt{context\_length}$ with $c$.

A token embedding is a vector with $\mathtt{embedding\_size}$ entries representing a token processed by the LLM; given $s \le c$ tokens, the input to the positional encoding layer is a matrix $X \in \mathbb{R}^{s \times e}$, where the $i$-th row of $X$ is the embedding of the $i$-th token. The positional encoding layer is then adding the $i$-th positional encoding vector to the embedding of the $i$-th token; in terms of matrix operation, the positional encoding layer is thus computing the output matrix $O \in \mathbb{R}^{s \times e}$ as:
$$
O[i, j] = X[i, j] + P[i, j]
$$
Therefore, only the first $s$ rows of the positional encoding matrix $P$ are involved in the computation.

### Proving Absolute Positional
The positional encoding matrix $P$ is a fixed matrix learnt during training. Being it independent from the inference input, the prover can commit to such matrix in a setup phase, and share the commitment with the verifier.

Despite the positional encoding layer is a simple matrix addition, which is trivial to prove, the main difficulty is that the addition is performed not with the committed matrix $P$, but with a sub-matrix $P_{\le s}$ given by the first $s$ rows of $P$. Therefore, the main issue is devising a protocol to provably bind an evaluation claim about matrix $P_{\le s}$ to an evaluation claim about the committed matrix $P$; indeed, the latter claim can be then proven to be bound to $P$ via an opening proof of the polynomial commitment scheme employed to commit to $P$.

#### Bind Claim About a Sub-Matrix

The main idea to devise a protocol to bind a claim about matrix $P_{\le s}$ to an evaluation claim about matrix $P$ is to split matrix $P$ row-wise in different chunks such that one of these chunks is exactly $P_{\le s}$, and then prove the proper split of matrix $P$ in these chunks.

To prove the proper split of a matrix in chunks, we rely on the following observation. Consider a matrix $M \in \mathbb{R}^{n \times m}$, where $n$ and $m$ are powers of 2. Assume to split the matrix in 2 chunks $M_1, M_2 \in \mathbb{R}^{\frac{n}{2} \times m}$, where $M_1$ is given by the first $\frac{n}{2}$ rows of $M$, and $M_2$ corresponds to the last $\frac{n}{2}$ rows of $M$. If we consider the MLEs of matrices $M_1, M_2$ and $M$, the following relantionship holds:
\begin{equation}
\small
M(x, y) = \beta(0, x_1)M_1(x_o, y) + \beta(1, x_1)M_2(x_o, y) = (1-x_1)M_1(x_o, y) + x_1M_2(x_o, y)
\tag{1}
\end{equation}
where $x_1$ is the variable referring to the most-significant bit of the rows in $M$ and $x_o$ are all the other variables related to the rows, i.e., $x = [x_1, x_o...]$. We can rely on this relationship to efficiently prove that $M_1$ and $M_2$ are the proper partition of $M$ as follows. 

Consider an evaluation claim $y_M = M(r_x, r_y)$, for random points $r_x \in \mathbb{F}^{\log_2(n)}$, $r_y \in \mathbb{F}^{\log_2(m)}$ chosen by the verifier. Consider point $r_x$ to be split in point $r_1 \in \mathbb{F}$ and point $r_o \in \mathbb{F}^{\log_2(n)-1}$. The prover computes the evaluation claims $y_1 = M_1(r_o, r_y)$, $y_2 = M_2(r_o, r_y)$, and sends these claims to the verifier. By checking that $y_M = (1-r_1)y_1 + r_1y_2$, the verifier is convinced that $y_1$ and $y_2$ are evaluations claims of the matrices $M_1$ and $M_2$, which are a proper partition of $M$.

Now, let's assume we need to provably derive a claim for a sub-matrix $M' \in \mathbb{R}^{\frac{n}{4} \times m}$ given by the first $\frac{n}{4}$ rows of $M$. By simply applying again the proving protocol we have just described, starting from the claim $y_1$ about $M_1$, we obtain claims $y_3$, $y_4$ about 2 sub-matrices $M_3$ and $M_4$, where $M_3 = M'$. Note that, as an optimization, there is no need for the prover to compute the claim $y_1$, but only the claims $y_3$ and $y_4$; indeed. the verifier can check that $y_3$ is bound to $y_M$ by checking the following relationship:
$$
y_M = (1-r_1)[(1-r_2)y_3 + r_2y_4] + r_1y_2
$$

Generalizing, if we need to provably derive a claim for a sub-matrix $M' \in \mathbb{R}^{\frac{n}{h} \times m}$ given by the first $\frac{n}{h}$ rows of $M$, where $h$ is a power of 2, we can recursively apply this proving protocol about $\log_2(h)$ times. In this way, it is possible to provably bind a claim of a matrix $M$ with a claim of a sub-matrix $M'$ given by a certain amount of consecutive rows of $M$.

#### How to Prove Positional Encoding
Equipped with this building block to bind a claim about a matrix $M \in \mathbb{R}^{n \times m}$ with a claim of the matrix $M' \in \mathbb{R}^{\frac{n}{h} \times m}$ given by the first $\frac{n}{h}$ rows of $M$, we now describe how to prove positional encoding layer.

First of all, we consider both the input matrix $X$ and the positional encoding matrix $P$ to be padded to the next power of 2 on each of their dimensions. That is, denoting the next power of 2 of an integer $n$ as $\hat{n}$, i.e, $\hat{n} = 2^{\lceil \log_2(n) \rceil}$, we consider the padded matrices $\hat{X} \in \mathbb{R}^{\hat{s} \times \hat{e}}$ and $\hat{P} \in \mathbb{R}^{\hat{c} \times \hat{e}}$. The output of the positional encoding layer would be the padded matrix $\hat{O} \in \mathbb{R}^{\hat{s} \times \hat{e}}$, where $\hat{O}[i,j] = \hat{X}[i, j] + \hat{P}[i, j]$.

The proving protocol starts from a claim $y_O = \hat{O}(r)$ about the output matrix $\hat{O}$, computed at a random point $r \in \mathbb{F}^{\log_2(\hat{s} \times \hat{e})}$ chosen by the verifier. The prover computes a claim $y_X = \hat{X}(r)$ and a claim $y_{P_s} = y_O - y_X$ about the matrix $\hat{P}_{\le \hat{s}} \in \mathbb{R}^{\hat{s} \times \hat{e}}$ given by the first $\hat{s}$ rows of matrix $\hat{P}$.

The prover then derives a claim $y_P$ for matrix $\hat{P}$ hinging upon the proving protocol described above. 
The prover will thus need to evaluate about $\log_2(h)$ MLEs related to sub-matrices of the positional matrix $\hat{P}$; note that the overall work to evaluate these MLEs is comparable to evaluating the MLE for the entire matrix $\hat{P}$, as the sum of the size of all such MLEs is basically the same as the matrix $\hat{P}$, but these evaluations can be computed in parallel by the prover. The $\log_2(h)$ claims will be part of the proof data sent to the verifier, and will be employed to re-compute the claim $y_P$ using recursively the relationship described in Equation 1.
The correctness of claim $y_P$ against the committed matrix $\hat{P}$ is then proven with an opening proof of the polynomial commitment scheme employed to commit to $\hat{P}$.

The claim $y_X$ is then returned as the claim for the input of the layer, which is the output of the proving protocol for the positional encoding layer.

## Rotary Positional Encoding
For a comprehensive overview of the RoPE layer, we defer the reader to [this blogpost](https://medium.com/@ngiengkianyew/understanding-rotary-positional-encoding-40635a4d078e).

For the sake of proving RoPE, we refer to the efficient way of computing RoPE, which is described in Property 2 in Part 4 of the above blogpost. 
More specifically, consider an input matrix $X \in \mathbb{R}^{s \times e}$, where $s$ denotes the number of tokens being processed and $e$ the $\mathtt{embedding\_size}$. Note that the number of tokens $s$ can be at most $c$, i.e., the $\mathtt{context\_length}$ of the LLM. 
To compute RoPE, we employ 2 matrices, defined as *rotational matrices*, which depends on a set of $\frac{e}{2}$ angles $\theta_1, \dots, \theta_{\frac{e}{2}}$ learnt during training.
More specifically, we build:

- a *cosine rotational matrix* $C \in \mathbb{R}^{c \times e}$ defined as $C[i,j] = cos(i*\theta_{\frac{j}{2}})$
- a *sine rotational matrix* $S \in \mathbb{R}^{c \times e}$ defined as $S[i,j] = sin(i*\theta_{\frac{j}{2}})$


The output tensor computed by rope is a tensor $O \in \mathbb{R}^{c \times e}$. The i-th row $O_i \in \mathbb{R}^e$ of $O$ is computed as follows. 

- Given a row $X_i \in \mathbb{R}^e = [x_1, x_2, \dots, x_e]$ of input matrix $X$, define the permuted row $\tilde{X}_i \in \mathbb{R}^e = [-x_2, x_1, \dots, -x_d, x_{d-1}]$, that is $\tilde{x}_i$ is obtained from $x_i$ by swapping pairs of consecutive elements and chaning the sign of the first element in the pair after the swap.
- Considering the corresponding rows $C_i$, $S_i$ of cosine matrix $C$ and sine matrix $S$, the row $O_i$ is computed as $O_i = X_i \otimes C_i + \tilde{X}_i \otimes S_i$, where $\otimes$ denotes the entry-wise tensor multiplication.

Therefore, only the first $s$ rows of rotational matrices $C$ and $S$ are employed in the RoPE computation, as for the positional matrix in absolute encoding.

### Proving RoPE
First of all, we assume that the input matrix $X$ and the rotational matrices $C$ and $S$ are already padded to the next power of 2. Therefore, recalling that we denote the next power of 2 of an integer $n$ as $\hat{n}$, i.e, $\hat{n} = 2^{\lceil \log_2(n) \rceil}$, we consider the padded matrices $\hat{X} \in \mathbb{R}^{\hat{s} \times \hat{e}}$, $\hat{C} \in \mathbb{R}^{\hat{c} \times \hat{e}}$ and $\hat{S} \in \mathbb{R}^{\hat{c} \times \hat{e}}$. The rotational matrices $\hat{C}$ and $\hat{S}$, being static tensors, are committed by the prover in a setup phase.

The proving protocol starts from an output claim $y_O$ about the MLE of the output matrix $\hat{O} \in \mathbb{F}^{\hat{s} \times \hat{e}}$, computed for a random point $r \in \mathbb{F}^{\log(\hat{s}) + \log(\hat{e})}$. We prove the computation of each row $\hat{O}_i \in \mathbb{F}^{\hat{e}}$ of $\hat{O}$ with a single sumcheck, which is basically applying the row-wise computation simultaneously on all the rows of the matrices.

To compute the sumcheck, the prover computes the following vectors:

- The $eq \in \mathbb{F}^{\hat{s} \times \hat{e}}$ vector for the claim $y_O$ defined as $eq[i] = \beta(r, i)$
- A pair-wise permutation $\widetilde{eq}$ of the vector $eq$ defined as
$$
\widetilde{eq}[i] = 
\begin{cases}
eq[i+1] \qquad \text{if} \quad i \bmod 2 == 0 \\
- eq[i-1] \quad \text{otherwise} 
\end{cases}
$$

The rationale behind the computation of these vectors is that the following relationship holds between the MLEs of these vectors and the MLE of the input matrix $\hat{X}$:

- $\sum_{i \in \{0,1\}^{\log(\hat{s}) + \log(\hat{e})}}eq(i)*X(i) = \hat{X}(r)$
- $\sum_{i \in \{0,1\}^{\log(\hat{s}) + \log(\hat{e})}}\widetilde{eq}(i)*X(i) = \widetilde{X}(r)$, where $\tilde{X}$ is the matrix obtained by transforming each row $\hat{X}_i$ of $\hat{X}$ to the permuted row $\widetilde{X}_i$

Intuitively, the vectors $eq$ and $\tilde{eq}$ are encoding the equality polynomials for the input matrix $\hat{X}$ and the permuted input matrix $\tilde{X}$.

Besides the MLEs for the vectors $eq$ and $\tilde{eq}$, the prover also builds the MLE of input matrix $\hat{X}$ and the MLEs of the sub-matrices $\hat{C}_{\le s}$ and $\hat{S}_{\le s}$ given by the first $s$ rows of matrices $\hat{C}$ and $\hat{S}$, respectively.
Then, the RoPE computation is proven with the following sumcheck:
$$
y_O = \sum_{i \in \{0,1\}^{\log(\hat{s}) + \log(\hat{e})}} eq(i)*\hat{X}(i)*\hat{C}_{\le s}(i) + \tilde{eq}(i)*\hat{X}(i)*\hat{S}_{\le s}(i)
$$
The sumcheck protocol yields the following claims, for a random point $r' \in \mathbb{F}^{\log(\hat{s}) + \log(\hat{e})}$:
- Claims $eq(r')$, $\tilde{eq}(r')$ about the equality polynomials, which can be efficiently verified by the verifier
- A Claim $\hat{X}(r')$ about the input matrix, which is the output of the proving protocol
- Claims $y_C = \hat{C}_{\le s}(r')$, $y_S = \hat{S}_{\le s}(r')$. These claims need to be bound to claims about the committed rotational matrices $\hat{C}$ and $\hat{S}$; this can be done by employing the sub-protocol described [above](#bind-claim-about-a-sub-matrix), which binds a claim about a matrix $M \in \mathbb{R}^{n \times m}$ with a claim of the matrix $M' \in \mathbb{R}^{\frac{n}{h} \times m}$ given by the first $\frac{n}{h}$ rows of $M$. The claims obtained by running this sub-protocol can be proven to be correct with an opening proof against the committed matrices $\hat{C}$ and $\hat{S}$ 