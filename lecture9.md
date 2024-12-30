## Lecture 9 

### Matrix Inversion is Unstable
- eg. Hilbert matrix: $A_{ij} = \frac{1}{i+j+1}$ 
	- $AA^{-1}$ should be $I$, does not get that in numpy
	- large condition number ($10^{17}$) 
- Solution: use other methods beside normal equation (QR, SVD, `lstqr`)
- Even if $A$ is sparse, $A^{-1}$ is generally dense: shouldn't take inverse

### LR runtimes
- Matrix inversion: $2n^3$
- Matrix Multiplication: $n^3$
- Cholesky: $\frac{1}{3}n^3 + 2n^2 \approx \frac{1}{3}n^3$
- QR, Gram Schmidt: $2mn^2, m \geq n$
- QR, Householder: $2mn^2 - \frac{2}{3}n^3$
- Solving triangular system: $n^2$
- Cholesky is fastest when it works (symmetric positive definite matrices), but is unstable for matrices with high condition numbers or low-rank)
- LR via QR is recommended as the standard method

### Additional SVD applications
- De-biasing Word2Vec word embeddings
	- Word2Vec: Google library that represents words as vectors
		- includes bias (eg. father:doctor :: mother:nurse)
- Data compression
- trades large number of features for smaller set of better features
- All matrices are diagonal if you change bases on the domain and range

### SVD vs Eigen Decomposition
- $A = U\Sigma V^T \Rightarrow Av_j = \sigma_j u_j$ 
	- generalization of eigen decomposition
	- eigen decomposition: SVD when $U = V$ ($Av = \lambda v$)
### Eigen Decomposition
- $AV = \Lambda V = V \Lambda$,  where $\lambda$ is a diagonal matrix with the eigenvalues
- Applications
	- rapid matrix powers
		- $A^k = (V\Lambda V^{-1})^k = V\Lambda^k V^{-1}$ 
		- $n^3\log(k)$ naive vs $3n^3 + n$ using eigen decomposition
	- Markov Chains
- Hermitian matrix: equal to its own conjugate transpose
	- All real symmetric matrices are Hermitian
- If $A$ is symmetric, then eigenvalues are real and $A = Q\Lambda Q^T$, $Q^{-1} = Q^T$
- If $A$ is triangular, then eigenvalues are equal to its diagonal entries

### Power Method
- Used by PageRank (Google) to rank importance of websites
- $n\times n$ matrix $A$ is diagonalizable if it has $n$ linearly independent eigenvectors $v_i$
	- then any $w$ can be expressed as $\sum^n_{j=1} c_jv_j$ 
	- $A^kw = \sum_j c_jA^kv_j = \sum_j c_j\lambda_j^kv_j$    
- Power method:
	- Normalize each column of $A$
	- Initialize $S = I$, normalize
	- Iterate:
		- $S := AS$, normalize
		- normalize when doing iterative calculations to prevent over/underflowing
- Many advanced eigenvalue algorithms are variations on the power method
	- Facebook's `fbpca` for PCA
- Convergence rate of the power method: ratio of the largest to second largest eigenvalue
	- Shifts: remove an eigenvalue to speed up convergence
	- Deflation: finds eigenvalues other than the largest
