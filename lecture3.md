## Lecture 3 
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
