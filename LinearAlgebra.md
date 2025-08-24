# Inverse of a 2×2 Matrix

Given the matrix

$$
A = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix},
$$

its inverse exists if and only if $\det(A) \neq 0$, where

$$
\det(A) = a_{11}a_{22} - a_{12}a_{21}.
$$

When $\det(A) \neq 0$, the inverse is

$$
A^{-1} = \frac{1}{\det(A)}\,\operatorname{adj}(A)
       = \frac{1}{\det(A)}
         \begin{bmatrix}
           a_{22} & -a_{12} \\
           -a_{21} & a_{11}
         \end{bmatrix}.
$$

## Adjugate and Cofactor Matrix

The adjugate (classical adjoint) of $A$ is the transpose of its cofactor matrix.

### Cofactors of a 2×2 Matrix

For $A = [a_{ij}]$, the cofactors are:

|  Entry   |    Sign factor    |  Minor   |  Cofactor   |
| :------: | :---------------: | :------: | :---------: |
| $a_{11}$ | $(-1)^{1+1} = +1$ | $a_{22}$ | $+\;a_{22}$ |
| $a_{12}$ | $(-1)^{1+2} = -1$ | $a_{21}$ | $-\;a_{21}$ |
| $a_{21}$ | $(-1)^{2+1} = -1$ | $a_{12}$ | $-\;a_{12}$ |
| $a_{22}$ | $(-1)^{2+2} = +1$ | $a_{11}$ | $+\;a_{11}$ |

For a $2\times2$ matrix:

-   Cofactor matrix

    $$
    C =
    \begin{bmatrix}
    \;\, a_{22} & -a_{21} \\
    -\, a_{12} & \;\, a_{11}
    \end{bmatrix}
    $$

-   Adjugate
    $$
    \operatorname{adj}(A) = C^\top =
    \begin{bmatrix}
    \;\, a_{22} & -a_{12} \\
    -\, a_{21} & \;\, a_{11}
    \end{bmatrix}
    $$

The inverse of matrix $A$ exists if and only if $\text{det}(A) \neq 0$. When $\text{det}(A) = 0$, the matrix is said to be **singular** or **non-invertible**.

---

### Properties of Matrix Inverses and Transposes

### Inverse Properties

1. **Identity Property**
   $$AA^{-1} = I = A^{-1}A $$

2. **Inverse of a Product**
   $$(AB)^{-1} = B^{-1}A^{-1} $$

3. **Inverse of a Sum (Inequality)**
   $$(A + B)^{-1} \neq A^{-1} + B^{-1} $$

### Transpose Properties

1. **Double Transpose**
   $$(A^{\top})^{\top} = A $$

2. **Transpose of a Product**
   $$(AB)^{\top} = B^{\top}A^{\top}$$

3. **Transpose of a Sum**
   $$(A + B)^{\top} = A^{\top} + B^{\top} $$

---

# Row Echelon Form (REF) and Gaussian Elimination

A matrix is in **row-echelon form (REF)** if:

1. All nonzero rows are above any rows of all zeros
2. In each nonzero row, the **first nonzero entry** (called the **leading entry** or **pivot**) is to the right of the leading entry in the row above
3. All entries **below** each pivot are zero

### Permitted Row Operations

These operations preserve the solution set:

-   **Row swap:** $R_i \leftrightarrow R_j$
-   **Scalar multiplication:** $R_i \leftarrow \lambda R_i$ where $\lambda \neq 0$
-   **Row addition:** $R_j \leftarrow R_j + \lambda R_i$

## 2. How to Compute REF (Gaussian Elimination)

Given an augmented matrix $[A \mid b]$:

1. **Choose a pivot column** (leftmost column with a nonzero entry below the current working row). If the top candidate is 0, **swap** with a lower row that has a nonzero entry there
2. **Create zeros below the pivot** using row replacements $R_j \leftarrow R_j + \lambda R_{\text{pivot}}$
3. Move one row down and one column right; **repeat** until every pivot has zeros beneath it
4. The resulting matrix is in **REF**. Solve by **back-substitution**

### Example: Solving a System by REF

### System of Equations

$$
\begin{aligned}
-2x_1 + 4x_2 - 2x_3 - x_4 + 4x_5 &= -3 \\
4x_1 - 8x_2 + 3x_3 - 3x_4 + x_5 &= 2 \\
x_1 - 2x_2 + x_3 - x_4 + x_5 &= 0 \\
x_1 - 2x_2 - 3x_4 + 4x_5 &= a
\end{aligned}
$$

### Augmented Matrix $[A \mid b]$

$$
\left[\begin{array}{ccccc|c}
-2 & 4 & -2 & -1 & 4 & -3 \\
4 & -8 & 3 & -3 & 1 & 2 \\
1 & -2 & 1 & -1 & 1 & 0 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}\right]
$$

### Step 1: Choose Pivot in Column 1

Swap $R_1 \leftrightarrow R_3$:

$$
\left[\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
4 & -8 & 3 & -3 & 1 & 2 \\
-2 & 4 & -2 & -1 & 4 & -3 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}\right]
$$

### Step 2: Zero Out Below Pivot in Column 1

Apply operations:

-   $R_2 \leftarrow R_2 - 4R_1$
-   $R_3 \leftarrow R_3 + 2R_1$
-   $R_4 \leftarrow R_4 - R_1$

Result:

$$
\left[\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & 1 & -3 & 2 \\
0 & 0 & 0 & -3 & 6 & -3 \\
0 & 0 & -1 & -2 & 3 & a
\end{array}\right]
$$

### Step 3: Pivot in Column 3

Use $R_2$ as pivot row for column 3. Apply $R_4 \leftarrow R_4 - R_2$:

$$
\left[\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & 1 & -3 & 2 \\
0 & 0 & 0 & -3 & 6 & -3 \\
0 & 0 & 0 & -3 & 6 & a-2
\end{array}\right]
$$

### Step 4: Final REF and Consistency Condition

Apply $R_4 \leftarrow R_4 - R_3$:

$$
\left[\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & 1 & -3 & 2 \\
0 & 0 & 0 & -3 & 6 & -3 \\
0 & 0 & 0 & 0 & 0 & a+1
\end{array}\right]
$$

This is in REF.

---

The last row encodes $0 = a+1$, so the system is consistent if and only if

$$
a = -1.
$$

Back‑Substitution (when a = −1)

With $a=-1$, discard the zero row. Optionally scale the pivot rows for easier reading:

-   $R_2 \leftarrow -R_2 \ \Rightarrow\ [0,0,1,-1,3\mid -2]$
-   $R_3 \leftarrow -\tfrac13 R_3 \ \Rightarrow\ [0,0,0,1,-2\mid 1]$

Pivot equations:

$$
\begin{aligned}
&x_4 - 2x_5 = 1 &&\Rightarrow&& x_4 = 1 + 2x_5,\\
&x_3 - x_4 + 3x_5 = -2 &&\Rightarrow&& x_3 = -2 + x_4 - 3x_5 = -1 - x_5,\\
&x_1 - 2x_2 + x_3 - x_4 + x_5 = 0 &&\Rightarrow&& x_1 = 2x_2 - x_3 + x_4 - x_5 = 2x_2 + 2 + 2x_5.
\end{aligned}
$$

    Free variables: Columns 2 and 5 have no pivots.

Let $x_2=t\in\mathbb{R}$, $x_5=s\in\mathbb{R}$.

Solution set (parametric form):

$$
(x_1,x_2,x_3,x_4,x_5)=\bigl(2t+2+2s,\ t,\ -1-s,\ 1+2s,\ s\bigr),\qquad s,t\in\mathbb{R}.
$$

-   The system effectively has only three independent equations (from the first three non-zero rows) for five variables ($ x_1, x_2, x_3, x_4, x_5 $).

-   The rank of the coefficient matrix $ A $ is at most 3 (since there are three non-zero rows), and the augmented matrix $ [A | \mathbf{b}] $ has the same rank because the last row contributes no new information.

-   With five variables and three independent equations, there are $ 5 - 3 = 2 $ free variables, which is why the solution has two parameters ($ s $ and $ t $) in the general solution

---

### Using REF to Compute a Particular Solution (Column-Space View)

The row-echelon form simplifies the process of determining a particular solution. To find a particular solution:

1. Express the right-hand side of the equation system using the pivot columns:

    $$b = \sum_{i=1}^{P} \lambda_i p_i$$

    where $p_i$ ($i = 1, \ldots, P$) are the pivot columns.

2. The coefficients $\lambda_i$ are determined most easily by:
    - Starting with the rightmost pivot column
    - Working systematically from right to left

-   Setting all free variables to zero produces a particular solution $x_p$ whose nonzero entries occur only in the pivot (basic) positions and equal the coefficients $\lambda_i$ above.

This approach leverages the triangular structure of the row-echelon form to solve for the coefficients through back-substitution, making the calculation process more straightforward and systematic.

In the previous example, we need to find $\lambda_1, \lambda_2, \lambda_3$ such that:

$$\lambda_1 \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix} + \lambda_3 \begin{bmatrix} -1 \\ -1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -2 \\ 1 \\ 0 \end{bmatrix} $$

### Solution Process

Working from right to left:

-   From the third row: $\lambda_3 = 1$
-   From the second row: $\lambda_2 - \lambda_3 = -2 \Rightarrow \lambda_2 = -1$
-   From the first row: $\lambda_1 + \lambda_2 - \lambda_3 = 0 \Rightarrow \lambda_1 = 2$

### Final Particular Solution

When assembling the complete solution, we must include the non-pivot columns (for which coefficients are implicitly set to 0). Therefore, the particular solution is:

$$\mathbf{x} = [2, 0, -1, 1, 0]^{\top}$$

where the zeros in positions 2 and 5 correspond to the non-pivot columns.

---

### Reduced Row-Echelon Form (RREF)

A matrix (or the coefficient matrix of a linear system) is in reduced row‑echelon form (also called row‑reduced echelon form or row canonical form) if:

1. It is in row‑echelon form (REF).
2. Every pivot (leading entry) equals $1$.
3. Each pivot is the only nonzero entry in its column.

♢ Gauss-Jordan elimination is an extension of Gaussian elimination that continues the process to produce reduced row-echelon form (RREF).

---

### Finding Solutions to Ax = 0 Using Pivot and Non-Pivot Columns

The key idea for finding the solutions of $A\mathbf{x} = \mathbf{0}$ is to look at the non-pivot columns, which we will need to express as a (linear) combination of the pivot columns.

Consider the matrix:

$$A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}$$

which is already in RREF.

#### 1. Identify Pivot and Non-Pivot Columns

Pivots appear at the first nonzero entry of each nonzero row:

- Row 1 pivot → column 1
- Row 2 pivot → column 3
- Row 3 pivot → column 4

Therefore:
- **Pivot columns:** 1, 3, 4
- **Non-pivot columns:** 2, 5

#### 2. Express Non-Pivot Columns as Linear Combinations of Pivot Columns

Write the column vectors (columns of $A$) explicitly:

$$\begin{aligned}
p_1 &= \text{col}_1(A) = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, &
p_2 &= \text{col}_2(A) = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}, \\
p_3 &= \text{col}_3(A) = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, &
p_4 &= \text{col}_4(A) = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \\
p_5 &= \text{col}_5(A) = \begin{bmatrix} 3 \\ 9 \\ -4 \end{bmatrix}.
\end{aligned}$$

Now express the non-pivot columns in terms of the pivots $p_1, p_3, p_4$:

##### Column 2:
$$p_2 = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix} = 3\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = 3p_1$$

##### Column 5:
$$p_5 = \begin{bmatrix} 3 \\ 9 \\ -4 \end{bmatrix} = 3\begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + 9\begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} - 4\begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = 3p_1 + 9p_3 - 4p_4$$

#### 3. Use This to Solve $A\mathbf{x} = \mathbf{0}$

Write $\mathbf{x} = (x_1, x_2, x_3, x_4, x_5)^T$. The homogeneous equation $A\mathbf{x} = \mathbf{0}$ is the vector equation:

$$x_1p_1 + x_2p_2 + x_3p_3 + x_4p_4 + x_5p_5 = \mathbf{0}$$

Substitute the expressions for $p_2$ and $p_5$:

$$x_1p_1 + x_2(3p_1) + x_3p_3 + x_4p_4 + x_5(3p_1 + 9p_3 - 4p_4) = \mathbf{0}$$

Collect coefficients of the pivot columns $p_1, p_3, p_4$:

$$(x_1 + 3x_2 + 3x_5)p_1 + (x_3 + 9x_5)p_3 + (x_4 - 4x_5)p_4 = \mathbf{0}$$

Because $p_1, p_3, p_4$ are linearly independent (they are standard basis vectors in $\mathbb{R}^3$), each coefficient must be zero:

$$\begin{cases}
x_1 + 3x_2 + 3x_5 = 0 \\
x_3 + 9x_5 = 0 \\
x_4 - 4x_5 = 0
\end{cases}$$

#### 4. Parameterize the Solution (Free Variables)

Non-pivot columns correspond to free variables. Let:

$$x_2 = s, \quad x_5 = t \quad (s, t \in \mathbb{R})$$

Then from the three equations above:

$$\begin{aligned}
x_1 &= -3x_2 - 3x_5 = -3s - 3t \\
x_3 &= -9x_5 = -9t \\
x_4 &= 4x_5 = 4t
\end{aligned}$$

So the general solution to $A\mathbf{x} = \mathbf{0}$ is:

$$\mathbf{x} = \begin{pmatrix} -3s - 3t \\ s \\ -9t \\ 4t \\ t \end{pmatrix} = s\begin{pmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{pmatrix} + t\begin{pmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{pmatrix}, \quad s, t \in \mathbb{R}$$

---
## The Null Space (Kernel) of a Matrix

The **null space** of a matrix $A$, denoted $\ker(A)$ (short for "kernel of $A$"), is the set of all vectors $\mathbf{x}$ in the domain of $A$ such that:

$$A\mathbf{x} = \mathbf{0}$$

### Mathematical Definition

$$\ker(A) = \{ \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \}$$

where $A$ is an $m \times n$ matrix, and $\mathbf{x}$ is an $n \times 1$ vector.

### Interpretation

The null space consists of all solutions to the homogeneous system of linear equations $A\mathbf{x} = \mathbf{0}$. These solutions represent vectors that, when multiplied by $A$, result in the zero vector. Geometrically, the null space describes the set of vectors that are "mapped to zero" by the linear transformation represented by $A$.

Consider the matrix:

$$A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}$$

This is a $3 \times 5$ matrix, so $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5]^T \in \mathbb{R}^5$, and the system $A\mathbf{x} = \mathbf{0}$ represents three equations with five unknowns. The null space $\ker(A)$ is the set of all vectors $\mathbf{x} \in \mathbb{R}^5$ that satisfy this system.

The general solution to $A\mathbf{x} = \mathbf{0}$ is:

$$\mathbf{x} = s \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + t \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix}, \quad s, t \in \mathbb{R}$$

### Basis for the Null Space

The null space $\ker(A)$ consists of all linear combinations of the vectors:

$$\left\{ \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix} \right\}$$

These two vectors form a **basis** for the null space, and the null space is **two-dimensional** because there are two free variables ($s$ and $t$).

## What Does $\ker(A)$ Represent?

### Algebraically

The null space contains all solutions to the homogeneous system. In this case, it describes all vectors $\mathbf{x}$ that satisfy:

$$\begin{cases}
x_1 + 3x_2 + 3x_5 = 0 \\
x_3 + 9x_5 = 0 \\
x_4 - 4x_5 = 0
\end{cases}$$

### Geometrically

If we think of $A$ as a linear transformation from $\mathbb{R}^5 \to \mathbb{R}^3$, the null space is the set of all vectors in $\mathbb{R}^5$ that are mapped to the zero vector in $\mathbb{R}^3$. In this case, $\ker(A)$ is a two-dimensional subspace of $\mathbb{R}^5$ (a "plane" in 5-dimensional space) spanned by the basis vectors above.

### Dimension of the Null Space

The dimension of $\ker(A)$, called the **nullity**, equals the number of free variables (non-pivot columns). Here:
- Total columns: 5
- Pivot columns: 3
- Non-pivot columns: $5 - 3 = 2$
- Therefore, nullity = 2

## Importance of the Null Space

### 1. Solutions to Homogeneous Systems

- If $\ker(A) = \{ \mathbf{0} \}$: only the trivial solution exists
- If $\ker(A)$ contains non-zero vectors: infinitely many solutions exist

### 2. Linear Independence

- If $\ker(A) = \{ \mathbf{0} \}$: columns of $A$ are linearly independent
- If $\ker(A)$ is non-trivial: columns of $A$ are linearly dependent

### 3. Rank-Nullity Theorem

$$\text{rank}(A) + \text{nullity}(A) = n$$

For our example:
- $\text{rank}(A) = 3$ (number of pivot columns)
- $\text{nullity}(A) = 2$ (dimension of null space)
- $n = 5$ (number of columns)

---
### Mechanical Method for Constructing Null Space Basis from RREF : The Free‑Variable Identity‑Block Pattern

This method leverages the structure of the RREF to systematically find the solutions to the homogeneous system $A\mathbf{x} = \mathbf{0}$.

## What is Done Here?

The process is a systematic algorithm to find a **basis for the null space** $\ker(A)$, which is the set of all vectors $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$. The method uses the RREF to identify:
- **Pivot columns** (corresponding to dependent variables)
- **Free columns** (corresponding to free variables)

Then constructs basis vectors by assigning values to free variables and solving for pivot variables.

## 1. Short Algorithm (Mechanical Recipe)

### Step-by-Step Method

1. **Identify pivot and free columns**
   - Pivot columns: those with a leading 1 in each non-zero row of RREF
   - Free columns: those without pivots (corresponding to free variables)

2. **Construct basis vectors**
   For each free variable:
   - Set that free variable to 1 and all other free variables to 0
   - For each pivot variable, set its value to the negative of the coefficient of the free variable in that pivot row
   - The resulting vector is a basis vector for the null space

3. **Form the basis matrix**
   - Collect coefficients of free variables in pivot rows into matrix $C$
   - Construct null space basis matrix $N = \begin{bmatrix} -C \\ I_k \end{bmatrix}$

## 2. Application to Example Matrix

Consider the matrix:

$$A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}$$

This is a $3 \times 5$ matrix in RREF. We seek all $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5]^T$ such that $A\mathbf{x} = \mathbf{0}$.

### Step 1: Identify Pivot and Free Columns

**Pivot columns:**
- Row 1, column 1 ($x_1$)
- Row 2, column 3 ($x_3$)
- Row 3, column 4 ($x_4$)

Therefore, pivot columns are 1, 3, and 4, corresponding to variables $x_1, x_3, x_4$.

**Free columns:** Columns 2 and 5 have no pivots, corresponding to free variables $x_2$ and $x_5$. There are $k = 2$ free variables.

### Step 2: Read Coefficients of Free Variables

The pivot rows give equations involving free variables $x_2$ and $x_5$:

- Row 1: $x_1 + 3x_2 + 3x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(3, 3)$
- Row 2: $x_3 + 9x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(0, 9)$
- Row 3: $x_4 - 4x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(0, -4)$

Collect these into matrix $C$:

$$C = \begin{bmatrix}
3 & 3 \\
0 & 9 \\
0 & -4
\end{bmatrix}$$

### Step 3: Construct Null Space Basis Matrix $N$

The null space basis matrix is:

$$N = \begin{bmatrix} -C \\ I_2 \end{bmatrix} = \begin{bmatrix}
-3 & -3 \\
0 & -9 \\
0 & 4 \\
1 & 0 \\
0 & 1
\end{bmatrix}$$

Reordering rows to match variable order $(x_1, x_2, x_3, x_4, x_5)$:

$$N = \begin{bmatrix}
-3 & -3 \\
1 & 0 \\
0 & -9 \\
0 & 4 \\
0 & 1
\end{bmatrix}$$

### Result: Basis Vectors

The columns of $N$ are the basis vectors:

$$v^{(1)} = \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad v^{(2)} = \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix}$$

These correspond to:
- For $x_2$: Set $x_2 = 1$, $x_5 = 0$, then $x_1 = -3$, $x_3 = 0$, $x_4 = 0$
- For $x_5$: Set $x_2 = 0$, $x_5 = 1$, then $x_1 = -3$, $x_3 = -9$, $x_4 = 4$

The null space is:

$$\ker(A) = \text{span} \left\{ \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix} \right\}$$

## 3. General Formula

For an $m \times n$ matrix in RREF with $r$ pivot columns and $k = n - r$ free columns:

### Setup
- Pivot indices: $i_1, i_2, \ldots, i_r$ (pivot variables)
- Free indices: $j_1, j_2, \ldots, j_k$ (free variables)

### Construction
1. Form the $r \times k$ matrix $C$:
   - Entry $C_{p,q}$ = coefficient of $q$-th free variable (column $j_q$) in $p$-th pivot row

2. Null space basis matrix:
   $$N = \begin{bmatrix} -C \\ I_k \end{bmatrix}$$

   where:
   - Top $r$ rows: pivot variables $x_{i_1}, x_{i_2}, \ldots, x_{i_r}$ (negated coefficients from $C$)
   - Bottom $k$ rows: free variables $x_{j_1}, x_{j_2}, \ldots, x_{j_k}$ (identity matrix $I_k$)
   - Rows arranged to match original variable order $x_1, x_2, \ldots, x_n$

Each column of $N$ is a basis vector for $\ker(A)$.

## Purpose and Advantages

### Why Use This Method?

1. **Mechanical**: No explicit back-substitution needed
2. **Systematic**: Works for any matrix in RREF
3. **Efficient**: Directly constructs basis vectors
4. **Clear structure**: The pattern reveals the relationship between free and pivot variables

### The "Free-Variable Identity-Block" Pattern

- **Identity block** $I_k$: Reflects setting one free variable to 1, others to 0
- **$-C$ block**: Ensures pivot variables satisfy $A\mathbf{x} = \mathbf{0}$
- **Standard technique**: Generalizes to any RREF matrix

This pattern makes the method a standard tool in linear algebra for finding null space bases efficiently and systematically.