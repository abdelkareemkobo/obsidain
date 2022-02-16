#linear_algebra #math #deeplearning 
# Why square matrices?
using a matrix,we can create an affine transformation.An affine transformation maps a set of points into another set of points so that points on a line in the original space are still on a line in the mapped space. The transfromation is 
y = Ax +b
The affine transform combines a matrix transform, A, with a translation, b, to
map a vector, x, to a new vector, y. We can combine this operation into a
single matrix multiplication by putting A in the upper-left corner of the matrix
and adding b as a new column on the right. A row of all zeros at the bottom
with a single 1 in the rightmost column completes the augmented
transformation matrix. For an affine transformation matrix
![[Pasted image 20220216001552.png]]
and translation vector 
![[Pasted image 20220216001633.png]]
we get 
![[Pasted image 20220216001641.png]]
This form maps a point, (x, y), to a new point, (x′, y′).
This maneuver is identical to the bias trick sometimes used when
implementing a neural network to bury the bias in an augmented weight
matrix by including an extra feature vector input set to 1.
***In fact, we can viewa feedforward neural network as a series of affine transformations, where the
transformation matrix is the weight matrix between the layers, and the bias vector provides the translation. The activation function at each layer alters the otherwise linear relationship between the layers. It is this nonlinearitythat lets the network learn a new way to map inputs so that the final output reflects the functional relationship the network is designed to learn*** we use square matrices then,to map points from one space back into the same space.
```ad-note
collapse: closed
A matrix doesn’t need to be square for NumPy to return the elements
along its diagonal. And although mathematically the trace generally only
applies to square matrices, NumPy will calculate the trace of any matrix,
returning the sum of the diagonal elements:
```
#### The identity matrix
NumPy uses np.identity or np.eye to generate identity matrices of a given size:
```python
>>> a = np.array([[1,2],[3,4]])
>>> i = np.identity(2)
>>> print(i)
[[1. 0.]
[0. 1.]]
>>> print(a @ i)
[[1. 2.]
[3. 4.]]
```
Look carefully at the example above. Mathematically, we said that
multiplication of a square matrix by the identity matrix of the same order
returns the matrix. NumPy, however, did something we might not want.
Matrix a was defined with integer elements, so it has a data type of int64, theNumPy default for integers. However, since we didn’t explicitly provide
np.identity with a data type, NumPy defaulted to a 64-bit float. Therefore,
matrix multiplication (@) between a and i returned a floating-point version of a. This subtle change of data type might be important for later calculations,so, again, we need to pay attention to data types when using NumPy.It doesn’t matter if you use np.identity or np.eye. In fact, internally, np.identity is just a wrapper for np.eye
#### Symmetric, Orthogonal, and Unitary Matrices
AA⊤ = A⊤ A = I
then A is an orthogonal matrix. If A is an orthogonal matrix, then
A−1 = A⊤
and, as a result,
det(A) = ±1

The definiteness of a matrix tells us something about the
eigenvalues, which we’ll learn more about in the next section. If a symmetric
matrix is positive definite, then all of its eigenvalues are positive. Similarly,
a symmetric negative definite matrix has all negative eigenvalues. Positive
and negative semidefinite symmetric matrices have eigenvalues that are all
positive or zero or all negative or zero, respectively
#### Vectors Norms and Distance Metrices
##### L∞-norm  or Chebyshev distance.
he maximum absolute value of the components of x.
If we replace x with the difference of two vectors, x − y, we can treat the
norms as distance measures between the two vectors. Alternatively, we canpicture the process as computing the vector norm on the vector that is the difference between x and y.
**Norm equations have other uses in deep learning. For example, weight
decay, used in deep learning as a regularizer, uses the L2-norm of the weightsof the model to keep the weights from getting too large. The L1-norm of the weights is also sometimes used as a regularizer**
##### Covariance matrix
This matrix captures the variance of the individual features along the main diagonal Meanwhile, the off-diagonal valuesrepresent how one feature varies as another varies—these are the
covariances
##### Mahalnobis Distance
Above, we represented a dataset by a matrix where the rows of the dataset
are observations and the columns are the values of variables that make up
each observation. In machine learning terms, the rows are the feature vectors.As we saw above, we can calculate the mean of each feature across all theobservations, and we can calculate the covariance matrix. With these values,we can define a distance metric called the Mahalanobis distance,
![[Pasted image 20220216013240.png]]
where x is a vector, μ is the vector formed by the mean values of each
feature, and Σ is the covariance matrix.
Equation 6.9 is, in some sense, measuring the distance between a vector
and a distribution with the mean vector μ. The dispersion of the distribution
is captured in Σ. If there is no covariance between the features in the dataset
and each feature has the same standard deviation, then Σ becomes the identity
matrix, which is its own inverse. In that case, Σ−1 effectively drops out of
Equation 6.9, and the Mahalanobis distance becomes the L2-distance
(Euclidean distance)
Another way to think of the Mahalanobis distance is to replace μ with
another vector, call it y, that comes from the same dataset as x. Then DM isthe distance between the two vectors, taking the variance of the dataset intoaccount
```ad-info
collapse: closed
We can use the Mahalanobis distance to build a simple classifier. If,
given a dataset, we calculate the mean feature vector of each class in the
dataset (this vector is also called the centroid), we can use the Mahalanobis
distance to assign a label to an unknown feature vector, x. We can do so by
calculating all the Mahalanobis distances to the class centroids and assigning
x to the class returning the smallest value. This type of classifier is
sometimes called a nearest centroid classifier, and you’ll often see it
implemented using the L2-distance in place of the Mahalanobis distance.
Arguably, you can expect the Mahalanobis distance to be the better metric
because it takes the variance of the dataset into account.
```
One recent use of the Mahalanobis distance in deep learning is to take the
top-level embedding layer values, a vector, and use the Mahalanobis
distance to detect out-of-domain or adversarial inputs. An out-of-domain
input is one that is significantly different from the type of data the model was
trained to use. An **adversarial** input is one where an adversary is
deliberately attempting to fool the model by supplying an input that isn’t of
class X but that the model will label as class X.
##### Kullback-leibler Divergence
The Kullback-Leibler divergence (KL-divergence), or relative entropy, is a
measure of the similarity between two probability distributions: the lower
the value, the more similar the distributions.
If P and Q are discrete probability distributions, the KL-divergence is
![[Pasted image 20220216013826.png]]
where log2 is the logarithm base-2. This is an information-theoretic measure;
the output is in bits of information. Sometimes the natural log, ln, is used, in
which case the measure is said to be in nats. The SciPy function that
implements the KL-divergence is in scipy.special as rel_entr. Note that rel_entr uses
the natural log, not log base-2. Note also that the KL-divergence isn’t a
distance metric in the mathematical sense because it violates the symmetryproperty, DKL(P||Q) ≠ DKL(Q||P), but that doesn’t stop people from using it
as one from time to time.
             

#### Singular value decomposition and pesudoinverse
you'll most likely encounter SVD when calculating the pseudoinverse of a nonnsquare matrix.
![[Pasted image 20220216014855.png]]
The SVD of an m × n matrix, A, returns the following: U, which is m × m
and orthogonal; Σ, which is m × n and diagonal; and V, which is n × n and
orthogonal. Recall that the transpose of an orthogonal matrix is its inverse, so
UU⊤ = Im and VV⊤ = In, where the subscript on the identity matrix gives the
order of the matrix, m × m or n × n
At this point in the chapter, you may have raised an eyebrow at the
statement “Σ, which is m × n and diagonal,” since we’ve only considered
square matrices to be diagonal. Here, when we say diagonal, we mean a
rectangular diagonal matrix. This is the natural extension to a diagonal
matrix, where the elements of what would be the diagonal are nonzero and
all others are zero. For example,
![[Pasted image 20220216015107.png]]
is a 3 × 5 rectangular diagonal matrix because only the main diagonal is
nonzero. The “singular” in “singular value decomposition” comes from the
fact that the elements of the diagonal matrix, Σ, are the singular values, the square roots of the positive eigenvalues of the matrix ATA.
- let's be explicit and use SVD to decompose a matrix.our text is 
![[Pasted image 20220216015233.png]]
```python
from scipy.linalg import svd
a = np.array([[3,2,2],[2,3,-2]])
u,s,vt = svd(a)
print(u,s,vt)
print(np.linalg.eig(a.T @ a)[0])
[2.5000000e+01 5.0324328e-15 9.0000000e+00]
```
This shows us that, yes, 5 and 3 are the square roots of 25 and 9. Recall thateig returns a list, the first element of which is a vector of the eigenvalues.Also note that there is a third eigenvalue: zero. You might ask: “How small anumeric value should we interpret as zero?” That’s a good question with nohard and fast answer. Typically, I interpret values below 10−9 to be zero.The claim of SVD is that U and V are unitary matrices. If so, their products with their transposes should be the identity matrix:
```python
print(u.T @ u)
[[1.00000000e+00 3.33066907e-16]
[3.33066907e-16 1.00000000e+00]]
print(vt @ vt.T)
[[ 1.00000000e+00 8.00919909e-17 -1.85037171e-17]
[ 8.00919909e-17 1.00000000e+00 -5.55111512e-17]
[-1.85037171e-17 -5.55111512e-17 1.00000000e+00]]
```
Given the comment above about numeric values that we should interpret as
zero, this is indeed the identity matrix. Notice that svd returned V⊤, not V.
However, since (V⊤)⊤ = V, we’re still multiplying V⊤V.
The svd function returns not Σ but the diagonal values of Σ. Therefore,
let’s reconstruct Σ and use it to see that SVD works, meaning we can use U,
Σ, and V⊤ to recover A
```python 
S = np.zeros((2,3))
S[0,0], S[1,1] = s
print(S)
[[5. 0. 0.]
[0. 3. 0.]]
A = u @ S @ vt
print(A)
[[ 3. 2. 2.]
[ 2. 3. -2.]]
```
This is the A we started with—almost: the recovered A is no longer of
integer type, a subtle change worth remembering when writing code.
###### Two applications
SVD is a cute trick, but what can we do with it? The short answer is “a lot.”
Let’s see two applications. The first is using SVD for PCA. The sklearn PCA
class we used in the previous section uses SVD under the hood. The second
example shows up in deep learning: using SVD to calculate the Moore-
Penrose pseudoinverse, a generalization of the inverse of a square matrix to
m × n matrices.
###### The moore-pensore psedoinverse
As promised, our second application is to compute A+, the Moore-Penrose
pseudoinverse of an m × n matrix A. The matrix A+ is called a pseudo-
inverse because, in conjunction with A, it acts like an inverse in that
![[Pasted image 20220216020218.png]]
where AA+ is somewhat like the identity matrix, making A+ somewhat like
the inverse of A.
Knowing that the pseudoinverse of a rectangular diagonal matrix is
simply the reciprocal of the diagonal values, leaving zeros as zero, followed
by a transpose, we can calculate the pseudoinverse of any general matrix as
A+ = VΣ+ U*
for A = UΣV*, the SVD of A. Notice, we’re using the conjugate transpose,
V*, instead of the ordinary transpose, V⊤. If A is real, then the ordinary
transpose is the same as the conjugate transpose.

