#linear_algebra 
- most of the effort in implementing highly performant deep learning toolkits invovles finding ways to do arithmetic whith tensors as efficiently as possilbe
### Vectors
- in deeplearning and machine learning in general,the components of a vector  are often unrelated to each other in any strict geometric sense.Rather,they're used to represent features,qualities of some sample that the model will use to attempt to arrive at a useful output,like a class label,or a regression value.
- you'll often hear deep learning people discuss the feature space of a problem.The feature space refers to the set of possible inputs.
...the training set for a model needs to accurately represent the feature space of the possible inputs the model will encounter when used.
in this sense,the feature vector s a point, a location  in this n-dimensional space where n is the number of features in the feature vector.
### matries
we can think of vectors as matrices with a single row or column.
A column vector with three element  is 3* 1 matrix: 
it has three rows and one column.similiary, a row vector of four elements acts like a 1 * 4 matrix.
it has one row and four columns.
### Tesnosrs
A scalar has no dimensions, a vector has one, and a matrix has two. As you
might suspect, we don’t need to stop there. A mathematical object with more
than two dimensions is colloquially referred to as a tensor. When necessary,
we’ll represent tensors like this: T, as a sans serif capital letter.
The number of dimensions a tensor has define its order,which is not to be confused with the order of a matrix .A 3D tensor has order 3.
A matrix is a tensor of order 2.
A vector is an order-1 tensor.
A scalar is an order-0 tensor
when we discuss the flow of data through a deep neural network ..we'll see that many toolkits use tensors of order 4(or more)
for example we can define an order-3 tensor as shown below
```python
t = np.arange(36).reshape((3,3,4))
print(t)
[[[ 0 1 2 3]
[ 4 5 6 7]
[ 8 9 10 11]]
[[12 13 14 15]
[16 17 18 19]
[20 21 22 23]]
[[24 25 26 27]
[28 29 30 31]
[32 33 34 35]]]
```
-Any tensor of less than order n can be represented as an order-n tensor
by supplying the missing dimensions of length one. We saw an example of
this above when I said that an m-component vector could be thought of as a 1
× m or an m × 1 matrix. The order-1 tensor (the vector) is turned into an
order-2 tensor (matrix) by adding a missing dimension of length one.
As an extreme example, we can treat a scalar (order-0 tensor) as an
order-5 tensor, like this
```python
t = np.array(42).reshape((1,1,1,1,1))
print(t)
[[[[[42]]]]]
t.shape
(1, 1, 1, 1, 1)
t[0,0,0,0,0]
42
```
we'll often use this trick of adding new dimensions of length one.in fact,in Numpy,there is a way to do this directly,,which you'll see when using deep learning toolkits.
```python
t = np.array()
print(t)
[[1 2 3]
[4 5 6]]
w = t[np.newaxis,:,:]
w.shape
(1, 2, 3)
print(w)
[[[1 2 3]
[4 5 6]]]
```
Here,we've turned t,an order-2 tensor (a matrix),into an order-3 tensor by using np.newaxis to create a new axis of length one.That's why w.shape returns  (1,2,3) and not (2,3), as it would for t.
# Arithmetic with Tensors
Element-wise multiplication on two matrices(a and b) is often known as the Hadamard product.in deep learning.
The numpy toolkit extends the idea of element-wise operations into what it calls **broadcasting**.When broadcasting,Numpy applies rules,which we'll see via examples,where one array is passed over another to produce a meaningful output.
#### projection
the projection of one vector onto another calculates the amount  of the first vector that's in the direction of the second.The projection of a onto b is 
![[Pasted image 20220215231935.png]]
Projection finds the component of a in the direction of b. Note the
projection of a onto b is not the same as the projection of b onto a.
Because we use an inner product in the numerator, we can see that the
projection of a vector onto another vector that’s orthogonal to it is zero. No
component of the first vector is in the direction of the second.
```python 
a = np.array([1,1])
b = np.array([1,0])
p = (np.dot(a,b)/np.dot(b,b))*b
print(p)
[1. 0.]
c = np.array([-1,1])
p = (np.dot(c,b)/np.dot(b,b))*b
print(p)
[-1. -0.]
```
#### Outer product
The outer product of two vectors instead returns a matrix.Note that unlike the inner product,the outer product does not require the two vectors to have the same number of components.
⊗ seems the most commonly used when the outer product
is presented with a binary operator.
```python
>>> a = np.array([1,2,3,4])
>>> b = np.array([5,6,7,8])
>>> np.dot(a,b)
70
>>> np.outer(a,b)
array([[ 5, 6, 7, 8],
[10, 12, 14, 16],
[15, 18, 21, 24],
[20, 24, 28, 32]])
```
The ability of the outer product to mix all combinations of its inputs has
been used in deep learning for neural collaborative filtering and visual
question answering applications. These functions are performed by advanced
networks that make recommendations or answer text questions about an
image. The outer product appears as a mixing of two different embedding
vectors. Embeddings are the vectors generated by lower layers of a network,
for example, the next to last fully connected layer before the softmax layer’s
output of a traditional convolutional neural network (CNN). The embedding
layer is usually viewed as having learned a new representation of the
network input. It can be thought of as mapping complex inputs, like images, to
a reduced space of several hundred to several thousands of dimensions.







#### Cross product
The cross product is widely used in physics and other sciences but is less
often used in deep learning because of its restriction to 3D space.
Nontheless, you should be familiar with it if you’re going to tackle the deep
learning literature.
This concludes our look at vector-vector operations. Let’s leave the 1D
world and move on to consider the most important operation for all deep
learning: matrix multiplication.
### Matrix multiplication 
The final form of matrix multiplication we’ll discuss is the Kronecker
product or matrix direct product of two matrices. When computing the
matrix product, we mixed individual elements of the matrices, multiplying
them together. For the Kronecker product, we multiply the elements of one
matrix by an entire matrix to produce an output matrix that is larger than the input matrices. The Kronecker product is also a convenient place to
introduce the idea of a block matrix, or a matrix constructed from smaller
matrices (the blocks).
![[Pasted image 20220215234911.png]]
we can define a block matrix, M, as the following.
![[Pasted image 20220215234920.png]]
where each element of M is a smaller matrix stacked on top of each other.
We can most easily define the Kronecker product using a visual example
involving a block matrix. The Kronecker product of A and B, typically
written as A ⊗ B, is
![[Pasted image 20220215234934.png]]
for A, an m × n matrix. This is a block matrix because of B, so, when writtenout completely, the Kronecker product results in a matrix larger than either A or B. Note, unlike matrix multiplication, the Kronecker product is defined for arbitrarily sized A and B matrices. For example, using A and B from Equation 5.10, the Kronecker product is
![[Pasted image 20220215235003.png]]
Notice above that we used ⊗ for the Kronecker product. This is the
convention, though the symbol ⊗ is sometimes abused and is used for other
things too. We used it for the outer product of two vectors, for example.
NumPy supports the Kronecker product via np.kron.


[[more in linear algebra]]