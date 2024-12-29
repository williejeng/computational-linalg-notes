# Computational Linear Algebra - Lecture Notes

https://github.com/fastai/numerical-linear-algebra/tree/master
https://www.stat.uchicago.edu/~lekheng/courses/309/books/Trefethen-Bau.pdf

## Lecture 1 (2024-12-21)

### Matrix Computations

#### Matrix & tensor products
- Matrix-vector products
- Matrix-matrix products
- Convolution, Correlation
#### Matrix decompositions
- NMF
- SVD
	- QR
- Pagerank (eigen decomposition)

### Accuracy
- Floating point arithmetic
- Conditioning & stability
	- How pertubations of the input impact the output
	- Classical vs Gram-Schmidt accuracy
	- Gram-Schmidt vs Householder (how orthogonal the answer is)
- Approximation accuracy
	- ex. bloom filters
		- no false negatives, some false positives

### Memory Use
- Sparse vs dense
	- finite memory problem

### Speed
- Computational complexity (Big-O)
- Vectorization
	- vectorized operations: apply to multiple elements at once on a single core
	- Low level linalg APIs
		-  BLAS, LAPACK
### Locality
- do computation required when in fast storage, access data stored next to each other

### Scalability / Parallelization


--- 
## Lecture 2 (2024-12-21)

### Singular Value Decomposition
- $A = U \Sigma V^T$
	- $A$: data matrix
		- hashtag/word co-occurence matrix, words as rows, hashtags as columns
	- $U$: left singular vectors, orthonormal
		- embedding of hashtags by the words they co-occur with
	- $\Sigma$: diagonal of singular values
	- $V^T$: right singular vectors, orthonormal
		- embedding of words by the hashtags they co-occur with
- SVD: exact decomposition
- Uses:
	- semantic analysis
	- collaborative filtering/recommendations (Netflix prize)
	- Moore-Penrose pseudoinverse
	- data compression
	- PCA
- Will come back to this

### Nonnegative Matrix Factorization - factorization of nonnegative data set $V$: $V = WH$
- positive factors are more easily interpretable
- NP-hard, non unique, non-exact
- applications
	- face decompositions
	- collaborative filtering
	- audio source separation
	- topic modeling

### Topic Frequency - Inverse Document Frequency (TF-IDF)
- normalizes term counts
- TF = (# of occurences of term t in document) / (# of words in documents)
- IDF = log(# of documents / # of documents with term t in it)

### Pytorch for NMF using SGD
- has `.cuda()` that uses GPU
- has `autograd` method that finds derivative

### Truncated SVD
- only calculates subset of columns 

### Randomized SVD
-  classical decomposition algorithms
	- big matrices
	- missing or inaccurate data
	- data transfer takes time, take advantage of GPUs
	- if $A$ is sparse or structured, use Krylov subspace methods, unstable
		- cost depends on properties of the input matrix and effort spent to stabilize
- advantages of randomized algorithms
	- inherently stable
	- don't depends on subtle spectral properties
	- matrix-vector products can be done in parallel
- to be continued

--- 
## Lecture 3 (2024-12-23)
### Randomized SVD
- Calculating truncated SVD
	1. Find $Q$ such that $A \approx Q Q^T A$ (approximation to the range of $A$), $Q$ has $r$ orthonormal columns
		1. Take random vectors $w_i$ , evaluate subspace formed by $A w_i$ 
		2. Form matrix $W$ with $w_i$ as its columns
		3. Then $AW = QR$ (QR decomposition), columns of $Q$ form orthonormal basis for $AW$, which is the range of $A$
		4. Since $AW$ has far more rows than columns, approximately orthonormal columns (due to probability)
	2. Construct $B = Q^T A$ , which is small $(r \times n)$
	3. Compute SVD of $B = S \Sigma V^T$ 
	4. Let $U = QS$, then $A \approx U \Sigma V^T$ 

### QR Decomposition
- $A=QR$, $Q$ has orthonormal columns, $R$ is upper triangular
- "most important idea in numerical linear algebra"

--- 
## Lecture 4 (2024-12-23)

### Randomized SVD
- way faster than full SVD
- Johnson-Lindenstrauss Lemma: small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved

### Principal Component Analysis (PCA)
- high-dimensional data has low intrinsic dimensionality
- classical PCA: seeks best rank-$k$ estimate $L$ of $M$
	- minimizing $\Vert M - L \Vert$ where $L$ has rank-$k$
	- brittle/sensitive with respect to grossly corrupted observations
- robust PCA: $M = L+S$, $L$ is low-rank, $S$ is sparse
	- low-rank: matrix has a lot of redundant information
	- sparse: captures corrupted entries
	- [Paper](https://arxiv.org/pdf/0912.3599)
	- applications:
		- video surveillance, face recognition
		- latent semantic indexing: $L$ captures common words, $S$ captures few key words that distinguish each document
		- ranking, collaborative filtering

### Background removal
#### With randomized SVD
- matrix $M$ ($r\times c$, $r$: number of pixels, $c$: number of frames)
- perform randomized SVD on $M$, lowering rank of $\Sigma$: columns become smoothed out, effectively removing background
#### With robust PCA
- low-rank: redundant info, background
- sparse: corrupted entities, foreground

### Robust PCA as optimization problem
- robust PCA can be written as: minimize $\Vert L \Vert_* + \lambda \Vert S \Vert_1$ subject to $L+S=M$
	- $\Vert \cdot \Vert_1$: L1 norm, minimizing results in sparse values, equal to maximum absolute column norm
	- $\Vert \cdot \Vert_*$: nuclear norm, L1 norm of the singular values, minimizing results in sparse singular values (low rank)

--- 
## Lecture 5 (2024-12-25)

### Robust PCA
- Algorithm: Principle component pursuit (pcp)
	1. Initialize $S_0 = Y_0$, $\mu > 0$ 
	2. While not converged do
		1. $L_{k+1} = D_\mu (M - S_k - \mu^{-1}Y_k)$ 
		2. $S_{k+1} = S_{\lambda\mu} (M - L_{k+1} - \mu^{-1}Y_k)$ 
		3. $Y_{k+1} = Y_k + \mu(M - L_{k+1} - S_{k+1})$ 
	3. Output $L, S$

### LU Factorization
- factors matrix into the product o a lower triangular matrix $L$ and upper triangular matrix $U$
- Using Gaussian Elimination
- Solving $Ax=b$ becomes $LUx=b$
	- solve $Ly=b$, then $Ux=y$ 
- We can store $L$ and $U$ by overwriting $A$, since the diagonal of $L$ is all 1's
	- doing factorizations in-place saves memory

### Stability
- Algorithm $\hat{f}$ for problem $f$ is stable if $\forall x, \frac{\Vert\hat{f}(x) - f(y) \Vert}{\Vert f(y)\Vert} = O(\epsilon_{machine})$  for some $y$ with $\frac{\Vert y-x\Vert}{\Vert x\Vert} = O(\epsilon_{machine})$
	- gives nearly the right answer to nearly the right question
- Backwards stability is stronger and simpler than stability
	- $\forall x, \hat{f}(x)=f(y)$ for some $y$ with $\frac{\Vert y-x\Vert}{\Vert x\Vert} = O(\epsilon_{machine})$ 
	- gives exactly the right answer to nearly the right question
- LU factorization is stable, but not backward stable
	- using floating point arithmetic, $A$ may not $= LU$
	- Solution: LU factorization with pivoting

### LU factorization with partial pivoting
- permute rows: at each step, choose largest value in column $k$, move that row to be row $k$
	- gets more stable answers
	- what is usually meant by LU factorization
- full pivoting: also permutes columns
	- significantly time consuming and rarely used in practice
- pivoting in practice using floating point arithmetic: can have exponentially growing numbers, lose precision
	- Let $PA=LU$ be computed by gaussian elimination with partial pivoting, then the computed matrices $\hat{P}, \hat{L}, \hat{U}$ satisfy $\hat{L}\hat{U} = \hat{P}A + \delta A,  \frac{\delta A}{A} = O(\rho\epsilon_{machine})$
		- $\rho$: growth factor, $\rho = \frac{max_{i,j} | u_{ij}|}{max_{i,j}|a_{ij}|}$ 
	- instability in gaussian elimination (with or without pivoting) arises only if $L$ and/or $U$ is large relative to the size of $A$
	- still stable in practice, hard to occur under natural circumstances

--- 
## Lecture 6 (2024-12-25)
### Randomized Projections
- If you take several random weighted averages of columns, you end up with columns that are not highly correlated to each other (roughly orthonormal) 

### Speeding up Gaussian Elimination
- LU decomposition can be fully parallelized
- Randomized LU is fully implemented to run on a standard GPU card without GPU-CPU data transfer

### `scipy.linalg.solve` vs `lu_solve`
- For problems with high growth factor, `lu_solve` may give the wrong answer 

### Block Matrices
- Ordinary Matrix Multiplication
	- Must read row of $A$ and column of $B$ into fast memory every time
	- $O(n^3)$
	- $n^2(1+n) = n^2+n^3$ reads, $n^2$ writes
- Block Matrix Multiplication
	- divide into $N\times N$ blocks of size $\frac{n}{N} \times \frac{n}{N}$, read block of $A, B$ at once
	- $O(n^3)$
	- $2N^3(\frac{n}{N})^2 = 2Nn^2$ reads, $n^2$ writes
		- less reads than ordinary multiplication ($N << n$), takes advantage of locality
	- more easily parallelized
	- more memory needed, can be optimized

### Broadcasting
- describes how arrays of different shapes are treated during arithmetic operations
- numpy broadcasting rules: dimensions are compatible if they are the same or one of them is 1

### Sparse matrices storage
- common storage formats
	- coordinate-wise (COO)
		- stores list of nonzero coordinates and values: (Val, Row, Col)
		- matrix-vector multiplication: $O(n)$
	- compressed sparse row (CSR)
		- stores lists of value, row-index range, column-index: (Val, Col), RowPtr
			- RowPtr_i = index of first nonzero element in row i
		- same # of operations for matrix-vector multiplication as COO
		- number of memory accesses reduced by 2
	- compressed sparse column (CSC)

___
## Lecture 7

---
##  Lecture 8

### Linear Regression
- Consider $X\beta = y$, where $X$ has more rows than columns (more data samples than variables)
	- We want to find $\hat{\beta}$ that minimizes $\Vert X\hat{\beta}-y\Vert_2$
- Polynomial features
	- adds more terms (second power)
	- use Numba to compile Python with C

### Regularization
- Lasso regression: L1 penalty

### Noise
- Add noise to the data
- Huber loss: less sensitive to outliers than squared error loss
	- quadratic for small error values, linear for large values
