#linear_algebra 
a nonzero row or column in a marrix means a row or column that contains at least one nonzero entry;.
**leading entry** of a row refers to the leftmost nonzero entry(in a nonzero rwo)

****
any nonzero matrix may be row reduced into more than one matrix in echlon form ,using different sequences of row operatoins.However,the reduced echelon form one obtains from a matrix is unique.
#### Theorem1: uniqueness of the reduced [[echelon form]]:
each matrix is row equivalent to one and only one reduced echelon matrix.
****
# [[ Pivot positions]]
when row operations on a matrix produce an echelon form,further row operations to obtain the reduced form do not change the positions of the leading entries.
Since the reduced echelon form is unique,the leading entires area always in the same positions in any echelon form obtained from a given matrix,These leading entries correspond to leading 1's in the reduced echelon form.
![[Pasted image 20220130164130.png]]
****
# Solutions of linear systems
![[Pasted image 20220130165540.png]]
![[Pasted image 20220130165756.png]]
# Parametric Descriptions of solution sets
the describtiosn in (5),(7)
![[Pasted image 20220130170208.png]]
![[Pasted image 20220130170224.png]]
are parametric descriptions of solution sets in which the free varaiables act as parameters.
Solving a system amounts to finding a parametric description of the solution set or determining that the solution set is empty..

Whenever a system is [[consistent]] and has free variables,the solution set has many parametric describtions.
![[Pasted image 20220130170652.png]]
****
Theorem2:**existence and uniqueness theorem**
A linear system is [[consistent]] if and only if the rightmost column of the augmented matrix is not a pivot column-that is, if and only if an echelon form of the augmented matrix has no row of the form 
[0 ...0 b] with b nonzero
if a linear system is consistent,then the solution set contains either  a unique solution,when there are no free variables,or infinitely many solutions ,when there is at least one free variable.
****
what is **[[interploating polynomail means]]?**

see example 33 and 34 in section 2 in chapter 1
