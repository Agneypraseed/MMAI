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

