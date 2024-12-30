## Lecture 5 

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