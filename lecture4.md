## Lecture 4 

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