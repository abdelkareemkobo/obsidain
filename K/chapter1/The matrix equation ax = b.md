#linear_algebra 
**remember**:
the presence of a free variable in a system doesn't guarantee that the system is consistent.
![[Pasted image 20220130220744.png]]
****
![[Pasted image 20220130221039.png]]
# Existence of solutions
the equation Ax = b has a solution if and only if b is a linear combination of the columns of A.
### Theorem 4
let A be an M * n matrix .Then the following statements are logically equivalent that is,for  a particular A,either they are all true statements or they are all flast. 
1. for each b in R$^m$,the equation Ax=b has a solution
2. each b in R$^m$,is a linear combination of the columns of A.
3. the columns of A span R$^m$,
4. A has a pivot position in every row.
```ad-warning
collapse: closed
therom 4 is about a coefficient matrix,not an agumented matrix. if an agumented matrix [A b ] has a pivot position in every row, then the equation Ax=b may or may not be consisten.
```
the augmented matrix could be inconsistent if it has a pivot in each row because \[1 2 3 0]\[0 0 0 2] 2 here is considered as pivot but there is contradiction so inconsistent.
****
theorem 5:
![[Pasted image 20220130225453.png]]

