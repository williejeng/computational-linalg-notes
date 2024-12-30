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

### Implementing Lineaer Regression
- `scikit.learn`
	- `scipy.linalg.lstqr`
		- LAPACK drivers:
			- `gelsd`: SVD, divide and conquer (default)
			- `gelsy`: QR factorization (can be faster)
			- `gelss`: SVD (used historically)
	- sparse version: `lsqr`
		- Uses Golub and Kahan bidiagonalization
- Naive solution: find $\hat{x}$ that minimizes $\Vert A\hat{x} - b \Vert_2$
	- find where $b$ is closest to the subspace spanned by $A$ (range of $A$)
		- projection of $b$ onto $A$
	- since $b-A\hat{x}$ must be perpendicular to the range of $A$, $A^T (b-A\hat{x}) = 0$ 
		- normal equations: $\hat{x} = (A^TA)^{-1}A^Tb$ 
		- If $A$ has full rank, then $(A^TA)^{-1}A^T$ is a square, hermitian positive definite matrix
			- can use Cholesky factorization
- Cholesky factorization: finds upper triangular $R$ s.t. $A^TA = R^TR$
	- Algorithm 11.1 from Trefethen
		- numpy and scipy give different upper/lower for Cholesky
	- $A^TAx = A^Tb \Rightarrow R^TRx=A^Tb \Rightarrow R^Tw=A^Tb \Rightarrow Rx=w$ 
		- solving linear system of equations for upper triangular matrix is easier
			- `scipy.linalg.solve_triangular`
- QR decomposition:
	- $Q$: orthonormal, $R$: upper triangular
	- $Ax=b \Rightarrow QRx = b \Rightarrow Rx = Q^Tb$ 
- SVD:  $Ax=b \Rightarrow U\Sigma Vx = b \Rightarrow \Sigma Vx=U^Tb \Rightarrow \Sigma w = U^Tb \Rightarrow x = V^Tw$ 
- Random Sketching Technique for Least Squares Regression
	1. Sample a $r\times n$ random matrix $S$, $r << n$
	2. Compute $SA, Sb$
	3. Find exact solution $x$ to regression $SA x = Sb$
	- https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/teaching/15859-fall19/scribe2.pdf

### Conditioning & stability
- Condition number: how small changes to the input cause the output to change
	- relative condition number: $\kappa = \sup_{\delta x} \frac{\Vert\delta f\Vert}{\Vert f(x)\Vert} / \frac{\Vert\delta x\Vert}{\Vert x\Vert}$ where $\delta x$ is infinitesimal 
		- $\kappa = 1, 10, 10^2$: problem is well-conditioned, $\kappa = 10^6, 10^{16}$: problem is ill-conditioned
	- condition number of a matrix: $\Vert A \Vert\Vert A^{-1} \Vert$ 
		- relates to computing $b$ or $x$ given $A$ and the other variable in $Ax=b$ 
- conditioning: perturbation behavior of a mathematical problem (eg. least squares)
	- eg. computing eigenvalues of a non-symmetric matrix: often ill-conditioned
- stability: perturbation behavior of an algorithm used to solve a problem on a computer (eg. least squares, gaussian elimination)