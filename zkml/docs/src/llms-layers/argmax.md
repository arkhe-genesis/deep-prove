# Argmax Layer
The argmax layer is responsible to select the set of next output token tokens predicted by an LLM model. More specifically, the argmax layer takes as input a matrix $X \in \mathbb{R}^{n \times v}$, where $n$ is the number of tokens being processed by the LLM and $v$ is the vocabulary size. Each row in the matrix $X$ is a vector of scores, where the i-th column is the score the i-th token in the vocabulary of the LLM. The argmax layer returns, for each row of the matrix $X$, the token in the vocabulary with the maximum score in that row. In other words, the layer is returning an output tensor $t \in \mathbb{Z}^n$ where $t[i] = argmax_j X[i][j]$

## Proving the Layer
Proving the argmax layer is done in 2 steps:

1. The prover builds the vector $m \in \mathbb{R}^n$ where $m[i] = max_j(X[i][j])$; in other words, $m[i]$ corresponds to the maximum score found in the i-th row of matrix $X$.
2. The prover then needs to prove that $X[i, t[i]] = m[i]$. 

### Step 1
In Step 1, the prover constructs the vector $m \in \mathbb{R}^n$ and commits to this vector. Then, the prover proves that $m[i] \ge X[i, j] \forall j \in \{0, \dots, v-1 \}$ by computing the matrix $D \in \mathbb{R}^{n \times v}$ such that $D[i,j] = m[i] - X[i,j]$, and range-check each entry of $D[i,j]$ to be non-negative via a lookup proof. The lookup proof yields a claim $y_D = D(r)$ for a random point $r \in \mathbb{F}^{\log(n) + \log(v)}$ chosen by the verifier. 

Consider now the point $r$ to be split in 2 sub-points:

- $r_1 \in \mathbb{F}^{\log(n)}$, which corresponds to the first $\log(n)$ coordinates of $r$
- $r_2 \in \mathbb{F}^{\log(v)}$, which corresponds to the remaining $\log(v)$ coordinates of $r$ 

The prover now evaluates the MLE of the vector $m$ on $r_1$, obtaining a claim $y_m = m(r_1)$; then, the prover computes a claim $y_X = X(r) = y_D - y_m$ for the MLE of the input matrix $X$. The correctness of claim $y_m$ is proven via an opening proof against the commitment of the vector $m$. 

Both the claims $y_m = m(r_1)$ and $y_X= X(r)$ are then employed in the second step of the proving protocol, which is described next.

### Step 2
The prover now, given the vector $m$ computed in step 1, needs to prove that $X[i, t[i]] = m[i]$. This is equivalent to proving that:

- $m[i]$ is found  in the i-th row of input matrix $X$, and since in step 1 it was proven that $m[i] \ge X[i,j] \forall j$, this basically proves that $m[i]$ is the maximum value in the i-th row of $X$
- $t[i]$ is the index of the column where such maximum value is located in the i-th row of input matrix $X$, and so by definition $t[i]$ is the argmax of the i-th row of $X$

To prove that $X[i, t[i]] = m[i]$, the prover employs the one-hot encoded matrix $\hat{t} \in \mathbb{Z}^{n \times v}$, defined from the output tensor $t \in \mathbb{Z}^n$ as 
$$
\hat{t}[i,j] = \begin{cases}
1 \quad \text{if} \quad t[i] == j \\
0 \quad \text{otherwise}  
\end{cases}
$$
Observe that the entry-wise multiplication of the input matrix $X$ with the one-hot encoded matrix $\hat{t}$ yields the matrix $\hat{M} \in \mathbb{R}^{n \times v}$ defined as:
$$
\hat{M}[i,j] = \begin{cases}
m[i] \quad \text{if} \quad t[i] == j \\
0 \quad \quad \text{otherwise}
\end{cases}
$$
Therefore, it holds that $\sum_{j=0}^{v-1} \hat{M}[i,j] = m[i], \forall i \in \{0, \dots, n-1 \}$, i.e., the sum of all the elements in the i-th row of $\hat{M}$ is exactly $m[i]$. These relationships are employed by the prover to prove with a single sum-check that $X[i, t[i]] = m[i]$.

More specifically, the prover starts from the claim $y_m = m(r_1)$ computed in step 1. Then, considering the point $r_s \in \mathbb{F}^{\log(n) + \log(v)} = r_1 || r_{inv}$, where $r_{inv} \in \mathbb{F}^{\log{v}} = [2^{-1}, \dots, 2^{-1}]$ is the vector with $\log{v}$ repetitions of the field element $2^{-1}$, the prover computes the vector $b \in \mathbb{F}^{nv}$ where $b[i] = \beta(i, r_s)$.
The prover then proves with sum-check the following relationship:
\begin{equation}
y_m = \sum_{i \in \lbrace 0, 1 \rbrace^{\log(n)+\log(v)}} 2^{\log(v)}b(i)X(i)\hat{t}(i)
\tag{1} 
\end{equation}
The sum-check produces the following claims, for a random point $r' \in \mathbb{F}^{\log(n) + \log(v)}$:

- Claim $b(r')$, which can be efficiently re-computed by the verifier
- Claim $y'_X = X(r')$, which is aggregated with the claim $y_X = X(r)$ computed in step 1, obtaining a single claim for the input tensor $X \in \mathbb{R}^{n \times v}$ 
- Claim $y_t = \hat{t}(r')$, which, because of the sparse structure of the one-hot encoded matrix $\hat{t} \in \mathbb{Z}^{n \times v}$, can be recomputed efficiently by the verifier in $O(n)$ time, given that the verifier knows the output tensor $t \in \mathbb{Z}^n$

#### Why The Sum-Check Relationship Holds
Consider the MLE of the sparse matrix $\hat{M} \in \mathbb{R}^{n \times v}$, the MLE of the input matrix $X \in \mathbb{R}^{n \times v}$, and the MLE of the one-hot encoded output matrix $\hat{t} \in \mathbb{Z}^{n \times v}$. 
For a generic point $z \in \mathbb{F}^{\log(n) + \log(v)}$, it holds that:
\begin{equation}
\hat{M}(z) = \sum_{j \in \lbrace 0, 1 \rbrace^{\log(n)+\log(v)}} \beta(j, z) X(j) \hat{t}(j)
\tag{2}
\end{equation}
Furthermore, given the MLE of the vector $m \in \mathbb{R}^n$, for a generic point $i \in \mathbb{F}^{\log(n)}$, it holds that:
\begin{equation} 
m(i) = 2^{\log(v)} \hat{M}(i, r_{inv})
\tag{3}
\end{equation} 
Indeed, by fixing the last $\log(v)$ variables of $\hat{M}$ to $r_{inv} \in \mathbb{F}^{\log(v)} = [2^{-1}, \dots, 2^{-1}]$, it can be checked that we obtain an MLE $\widetilde{m}(x)$, for $x \in \mathbb{F}^{\log(n)}$, defined as:
$$
\widetilde{m}(x) = \sum_{i \in \{0,1\}^{\log(n)}} \beta(i, x) \frac{\sum_{j \in \{0,1\}^{\log(v)}} \hat{M}[i, j]}{2^{\log(v)}} 
$$
Therefore, by multiplying $\widetilde{m}$ by the constant factor $2^{\log(v)}$, we obtain the MLE of the vector $m$.

Wrapping up, the sum-check relation in (1) is thus computing: 
$$
\sum_{i \in \lbrace 0, 1 \rbrace^{\log(n)+\log(v)}} 2^{\log(v)}b(i)X(i)\hat{t}(i) = 2^{\log(v)}\sum_{i \in \lbrace 0, 1 \rbrace^{\log(n)+\log(v)}} \beta(i, r_s)X(i)\hat{t}(i) = 2^{\log(v)}\hat{M}(r_s) 
$$
where the last step derives from the MLE relationship in (2). Given that $r_s = (r_1, r_{inv})$, then the evaluation $2^{\log(v)}\hat{M}(r_s) = 2^{\log(v)}\hat{M}(r_1, r_{inv})$, which, according to (3), is equivalent to $y_m = m(r_1)$  

