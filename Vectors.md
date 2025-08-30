The unit vector of $a$ is:

$$
\hat{a} = \frac{a}{\|a\|}
$$

---

The angle $\theta$ between two vectors $a$ and $b$ is found using the **dot product formula**:

$$
a \cdot b = \|a\| \, \|b\| \, \cos \theta.
$$

$$
\cos \theta = \frac{a \cdot b}{\|a\| \, \|b\|} 
\quad \Rightarrow \quad
\theta = \arccos \left( \frac{a \cdot b}{\|a\| \, \|b\|} \right).
$$


---

The projection of vector $a$ onto vector $b$ (denoted $\text{proj}_b(a)$) is given by:

$$
\text{proj}_b(a) = (a \cdot \hat{b}) \, \hat{b}
$$

$a \cdot \hat{b}$ is just a **scalar** (the component of $a$ along the direction of $b$).

Multiplying back by $\hat{b}$ makes it a **vector** in the direction of $b$.


$$
\text{proj}_b(a) = \frac{a \cdot b}{\|b\|^2} \, b.
$$

---
### Cross Product

### 1. Cross Product of Two Vectors: $\mathbf{a} \times \mathbf{b}$

The cross product is a vector operation that produces a vector perpendicular to both input vectors. This operation is **only defined for 3D vectors**.

The magnitude of the cross product is given by:

$$\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\| \|\mathbf{b}\| \sin\theta$$

where $\theta$ is the angle between vectors $\mathbf{a}$ and $\mathbf{b}$.

The direction is determined by the **right-hand rule**: curl your fingers from $\mathbf{a}$ to $\mathbf{b}$, and your thumb points in the direction of $\mathbf{a} \times \mathbf{b}$.

#### Component Form Computation

If $\mathbf{a} = (a_1, a_2, a_3)$ and $\mathbf{b} = (b_1, b_2, b_3)$, then:

$$\mathbf{a} \times \mathbf{b} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}$$

Expanding the determinant:

$$\mathbf{a} \times \mathbf{b} = (a_2b_3 - a_3b_2, \, a_3b_1 - a_1b_3, \, a_1b_2 - a_2b_1)$$

### Cross Product of Three Vectors: $\mathbf{a} \times (\mathbf{b} \times \mathbf{c})$

This operation is called the **vector triple product**.

### Important Properties

- **Non-associative**: $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) \neq (\mathbf{a} \times \mathbf{b}) \times \mathbf{c}$
- The result lies in the plane of $\mathbf{b}$ and $\mathbf{c}$, not perpendicular like a simple cross product

### BAC-CAB Rule

The vector triple product can be computed using the formula:

$$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = (\mathbf{a} \cdot \mathbf{c})\mathbf{b} - (\mathbf{a} \cdot \mathbf{b})\mathbf{c}$$

This formula is commonly remembered as the "BAC-CAB rule" based on the arrangement of vectors in the final expression.

---
#### Area of Parallelogram Spanned by $\mathbf{a}$ and $\mathbf{b}$

$$\text{Area} = \|\mathbf{a} \times \mathbf{b}\| $$

- Equivalent dot-product form (no cross product needed):
$$
\|\mathbf{a}\times\mathbf{b}\| \;=\; \sqrt{\|\mathbf{a}\|^2\|\mathbf{b}\|^2 - (\mathbf{a}\cdot\mathbf{b})^2}.
$$

- Scalar triple product :
$$
\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}) \;=\;
\det\!\begin{bmatrix}
a_1 & b_1 & c_1\\
a_2 & b_2 & c_2\\
a_3 & b_3 & c_3
\end{bmatrix}.
$$

- The sign indicates the orientation of the vectors. A positive result signifies a right-handed system, while a negative result indicates a left-handed one.

#### Volume of Parallelepiped Spanned by $\mathbf{a}$, $\mathbf{b}$, $\mathbf{c}$

$$\text{Volume} = |\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})|$$

This is the **absolute value** of the **scalar triple product**.

-  If the scalar triple product of three non-zero vectors is zero, the vectors are coplanar

#### Equivalent Determinant Form

$$\text{Volume} = \left|\det\begin{bmatrix} a_1 & b_1 & c_1 \\ a_2 & b_2 & c_2 \\ a_3 & b_3 & c_3 \end{bmatrix}\right|$$

where the columns (or rows) are the vectors.

- The scalar triple product is cyclic: $\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = \mathbf{b} \cdot (\mathbf{c} \times \mathbf{a}) = \mathbf{c} \cdot (\mathbf{a} \times \mathbf{b})$

---
### Gram Matrix

Given vectors $v_1,\dots,v_k \in \mathbb{R}^n$, the Gram matrix $G \in \mathbb{R}^{k\times k}$ is
$$
G = \begin{bmatrix}
v_1\!\cdot\!v_1 & v_1\!\cdot\!v_2 & \cdots & v_1\!\cdot\!v_k\\
v_2\!\cdot\!v_1 & v_2\!\cdot\!v_2 & \cdots & v_2\!\cdot\!v_k\\
\vdots & \vdots & \ddots & \vdots\\
v_k\!\cdot\!v_1 & v_k\!\cdot\!v_2 & \cdots & v_k\!\cdot\!v_k
\end{bmatrix}.
$$
Matrix form: if $V=[\,v_1\ \cdots\ v_k\,]\in\mathbb{R}^{n\times k}$, then
$$
G = V^\top V.
$$

The squared $k$‑dimensional volume of the parallelepiped spanned by $v_1,\dots,v_k$ is
$$
\operatorname{Vol}_k^2(v_1,\dots,v_k) = \det(G).
$$
Equivalently,
$$
\operatorname{Vol}_k(v_1,\dots,v_k) = \sqrt{\det(G)}.
$$

#### Properties

1. **Symmetric**: $G^{\top} = G$ (since $\mathbf{v}_i \cdot \mathbf{v}_j = \mathbf{v}_j \cdot \mathbf{v}_i$)
2. **Positive semidefinite**: All eigenvalues are non-negative
3. **Rank**: $\text{rank}(G) = \text{rank}(V)$ = dimension of span of the vectors

$$\det(G) = \text{(k-dimensional volume)}^2$$

of the parallelepiped spanned by $\mathbf{v}_1, \ldots, \mathbf{v}_k$.

The Gram determinant formula works in **any dimension** and for **any number of vectors**, making it a powerful generalization of cross product-based volume formulas.

If $V = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k]$ is the $n \times k$ matrix with your vectors as columns, then the Gram matrix is:

$$G = V^{\top}V$$

So $G$ is $k \times k$.

$$\det(G) = \det(V^{\top}V)$$

- When $V$ is **square**: $\det(V^{\top}V) = (\det V)^2$
- When $V$ is **rectangular** ($n \times k$ with $n \geq k$): $\det(V^{\top}V)$ is still the square of the product of the singular values of $V$

Therefore:

$$\det(G) = (\text{volume}_k)^2$$

#### In 2D
For vectors $\mathbf{a}, \mathbf{b}$:

$$\det(G) = (\mathbf{a} \cdot \mathbf{a})(\mathbf{b} \cdot \mathbf{b}) - (\mathbf{a} \cdot \mathbf{b})^2$$

This simplifies to:
$$= \|\mathbf{a}\|^2 \|\mathbf{b}\|^2 - (\mathbf{a} \cdot \mathbf{b})^2 = \|\mathbf{a}\|^2 \|\mathbf{b}\|^2 (1 - \cos^2\theta)$$

$$= (\|\mathbf{a}\|\|\mathbf{b}\|\sin\theta)^2 = \text{(area)}^2$$

#### In 3D
  $$
  \det(G)=\bigl(v_1\cdot(v_2\times v_3)\bigr)^2
  =\bigl(\text{parallelepiped volume}\bigr)^2.
  $$

---
