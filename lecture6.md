## Lecture 6
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