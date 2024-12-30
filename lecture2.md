## Lecture 2 

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