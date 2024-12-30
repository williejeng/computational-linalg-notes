## Lecture 1 

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
