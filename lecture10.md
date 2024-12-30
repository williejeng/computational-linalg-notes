## Lecture 10 

### QR Algorithm
- finds all eigenvalues of a symmetric positive definite matrix
- Linalg review
	- $A,B$ are similar if there exists non-singular matrix $X$ st. $B = X^{-1}AX$
		- change of basis
		- $A$ and $X^{-1}X$ have the same eigenvalues
	- Schur factorization of matrix $A$ is $A=QTQ^*$
		- $Q$ is unitary, $T$ is upper-triangular
			- Unitary: $QQ^* = Q^*Q = I$ 
		- the eigenvalues of a triangular matrix are its diagonal
			- $A$'s eigenvalues are the diagonal of $T$
		- Every square matrix has a Schur factorization
- Algorithm
	-  pure QR algorithm 
		1. $A_0 = A$
		2. Until convergence:
			1. $Q_k, R_k = A_{k-1}$
			2. $A_k = R_kQ_k$
		3. $A_k$ is the Schur form of $A$ 
		- why it works
			- $A_{k-1} = Q_kR_k \Rightarrow Q_k^{-1}A_{k-1} = R_k$
			- $R_kQ_k = Q_k^{-1}A_{k-1}Q_k \Rightarrow A_k = Q^{-1}_k\dots Q^{-1}_1 A Q_1 \dots Q_k$
			- $A^k = Q_1\dots Q_kR_k \dots R_1$  (proved in Trefethen p. 216-217)
		- converges very slow, not guaranteed to converge
	- practical QR algorithm
		- adds shifts
			- Instead of factoring $A_k$, we factor $A_k - s_k I = Q_kR_k$
			- Set $A_{k+1} = R_kQ_k + s_kI$ 
			- $s_k$: approximates an eigenvalue of $A$, we'll use $s_k = A_k(m,m)$
			- adding shifts speeds up convergence in many algorithms
				- power method, inverse iteration, Rayleigh quotient iteration
		- $O(n^4)$ to converge, $O(n^3)$ for symmetric matrices
			- Hessenberg matrix (zeros below the first subdiagonal): $O(n^3), O(n^2)$
	- Two-phase approach (used in-practice)
		- reduce matrix to Hessenberg form, then QR algorithm 

### Implementing QR Factorization
- linalg review: projections
	-   orthogonality: the line from $b$ to $p$ ($b-\hat{x}a$) is perpendicular to $a$
		- $a\cdot (b-\hat{x}a) = 0 \Rightarrow \hat{x} = \frac{a\cdot b}{a\cdot a}$ 
- Classical Gram-Schmidt
	- for each $j$, calculate single projection $v_j = P_j a_j$, where $P_j$ projects onto the space orthogonal to the span of $q_1, \dots , q_{j-1}$ 
	- $a_1 = r_{11}q_1, a_2 = r_{12}q_1 + r_{22}q_2, \dots$ 
	- unstable
- Modified Gram-Schmidt
	- for each $j$, calculate $j-1$ projections $P_j = P_{\perp q_{j-1} \dots \perp q_2 \perp q_1}$ 
- Householder: Orthogonal Triangularization $Q_n \dots Q_2 Q_1A = R$ 
	- Householder reflectors
	- (do more independent research)
- Modified vs Classic Gram-Schmidt graph
	- place where it levels off: $\epsilon_{machine}$ 
- `np.linalg.qr`: householder