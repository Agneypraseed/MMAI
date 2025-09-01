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

### General Line

Any line in 2D can be written as:

$$Ax + By + C = 0$$

### Normal Vector

The **normal vector** is simply:

$$\mathbf{n} = \langle A, B \rangle$$

because the coefficients $A, B$ are exactly the components perpendicular to the line.

Take any two distinct points on the line, $P_1=(x_1,y_1)$ and $P_2=(x_2,y_2)$, so
$$
Ax_1 + By_1 + C = 0, \qquad Ax_2 + By_2 + C = 0.
$$
The vector along the line from $P_1$ to $P_2$ is
$$
\mathbf{v} = \langle x_2 - x_1,\ y_2 - y_1\rangle.
$$
Compute the dot product with $\mathbf{n}=\langle A,B\rangle$:
$$
\mathbf{n}\cdot\mathbf{v} = A(x_2-x_1) + B(y_2-y_1).
$$
Subtracting the two line equations gives
$$
\big(Ax_2 + By_2 + C\big) - \big(Ax_1 + By_1 + C\big) = 0
\;\;\Longrightarrow\;\;
A(x_2-x_1) + B(y_2-y_1) = 0.
$$
Thus $\mathbf{n}\cdot\mathbf{v}=0$ for every direction vector $\mathbf{v}$ along the line, so $\mathbf{n}$ is perpendicular to the line.

#### The Gradient Explanation

This connects to a profound principle:

$$\nabla f = \langle A, B \rangle \text{ is } \perp \text{ to level curves of } f(x,y) = Ax + By + C$$

**Broader principle**: This generalizes beautifully to:
- 3D: The normal to plane $Ax + By + Cz + D = 0$ is $\langle A, B, C \rangle$
- n-D: The normal to hyperplane $\sum a_i x_i + c = 0$ is $\langle a_1, a_2, \ldots, a_n \rangle$

### Direction Vector

A **direction vector** is any vector perpendicular to $\mathbf{n}$. One easy choice:

$$\mathbf{d} = \langle -B, A \rangle$$

### Slope of a Line
For a line in slope–intercept form
$$
y = m x + c,
$$
the slope is $m = \tan\theta$, where $\theta$ is the angle the line makes with the $x$-axis.

Rewrite as:
$$-mx + y - c = 0$$

So $A = m$, $B = -1$.

- **Normal**: $\mathbf{n} = $ $\langle -m, 1 \rangle$
- **Direction**: $\mathbf{d} = \langle 1, m \rangle$ 

#### Angle Between Two Lines
For lines with slopes $m_1$ and $m_2$, the angle $\phi$ between them satisfies
$$
\tan \phi = \left|\frac{m_1 - m_2}{1 + m_1 m_2}\right|.
$$
If the lines are perpendicular, then $\phi = 90^\circ$, so $\tan \phi$ is undefined. This occurs exactly when the denominator is zero:
$$
1 + m_1 m_2 = 0 \quad \Longrightarrow \quad m_1 m_2 = -1.
$$

- Vertical line $x = a$ has undefined slope; a perpendicular line is horizontal $y = b$ (slope $0$).

---
### Planes 

The general form of a plane in 3D is:

$$Ax + By + Cz + D = 0$$

- Distance from point $(x_0, y_0, z_0)$ to plane:
  $$d = \frac{|Ax_0 + By_0 + Cz_0 + D|}{\sqrt{A^2 + B^2 + C^2}}$$

The vector

$$\mathbf{n} = \langle A, B, C \rangle$$

is **perpendicular to the plane**.

A **direction vector** is any vector **lying in the plane**.

Direction vectors lie in the plane and are therefore orthogonal to the normal:
$$
\mathbf{u}\cdot\mathbf{n}=0,\quad \mathbf{v}\cdot\mathbf{n}=0.
$$
Equivalently, solve the homogeneous equation
$$
Ax + By + Cz = 0
$$
to obtain two independent solutions; these give two independent direction vectors $\mathbf{u},\mathbf{v}$.

#### Point–normal and parametric forms
- Point–normal form (given a point $P_0=(x_0,y_0,z_0)$ on the plane):
  $$
  \mathbf{n}\cdot\big(\langle x,y,z\rangle - \langle x_0,y_0,z_0\rangle\big)=0.
  $$
- Parametric form (using two independent direction vectors $\mathbf{u},\mathbf{v}$):
  $$
  \mathbf{r}(s,t)=\mathbf{p}_0 + s\,\mathbf{u} + t\,\mathbf{v},\qquad s,t\in\mathbb{R},
  $$
  where $\mathbf{p}_0$ is any point on the plane.

- Projection of a point onto the plane

  With signed distance $d=\dfrac{Ax_0+By_0+Cz_0+D}{\sqrt{A^2+B^2+C^2}}$ and unit normal $\hat{\mathbf{n}}$,
  $$
  \mathrm{proj}_{\text{plane}}(x_0,y_0,z_0)
  = \langle x_0,y_0,z_0\rangle - d\,\hat{\mathbf{n}}.  
  $$

---



