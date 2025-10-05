## Determinants

The determinant is a scalar value that encodes both the **volume** and **orientation** of the geometric object (parallelepiped) formed by the rows or columns of a matrix.

#### Determinant in Two Dimensions (Area)

Suppose we have two vectors in $\mathbb{R}^2$:

$$\mathbf{a} = (a_1, a_2), \quad \mathbf{b} = (b_1, b_2)$$

If we put these as rows (or columns) of a $2 \times 2$ matrix:

$$A = \begin{bmatrix} a_1 & a_2 \\ b_1 & b_2 \end{bmatrix}$$

then the determinant is:

$$\det(A) = a_1 b_2 - a_2 b_1$$

### Geometric Interpretation

-   The parallelogram spanned by $\mathbf{a}$ and $\mathbf{b}$ has **area** $|\det(A)|$
-   The **sign** of $\det(A)$ indicates orientation:
    -   **Positive**: if $\mathbf{a}$ rotated counterclockwise to $\mathbf{b}$
    -   **Negative**: if clockwise

So in 2D, the determinant is the **signed area** of the parallelogram formed by the two vectors.

#### Determinant in Three Dimensions (Volume)

Consider three vectors in $\mathbb{R}^3$:

$$\mathbf{a}, \mathbf{b}, \mathbf{c}$$

If we arrange them as rows of a $3 \times 3$ matrix $A$, then:

$$\det(A) = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$$

-   $|\det(A)|$ gives the **volume** of the parallelepiped formed by $\mathbf{a}, \mathbf{b}, \mathbf{c}$

#### General Case in $\mathbb{R}^n$

For an $n \times n$ matrix:

-   $|\det(A)|$ gives the $n$-dimensional volume (hypervolume) of the parallelepiped spanned by the rows (or columns)
-   The sign encodes orientation relative to the standard basis

For any square matrix $A$:
$$\boxed{\text{Signed Volume} = \det(A)}$$

---

### Row Operations and Determinants

**Row operations preserve the linear dependency relationships between the columns.**

### 1. Row Swap

**Swapping two rows** multiplies the determinant by $-1$.

$$R_i \leftrightarrow R_j \implies \det(A') = -\det(A)$$

### 2. Row Addition

**Adding a multiple of one row to another** does not change the determinant.

$$R_i \leftarrow R_i + kR_j \implies \det(A') = \det(A)$$

### 3. Row Scaling

**Multiplying a row by a non-zero scalar $k$** multiplies the determinant by $k$.

$$R_i \leftarrow kR_i \implies \det(A') = k\cdot\det(A)$$

#### Example

Let's compute the determinant of:

$$
A = \begin{bmatrix}
2 & 1 & 3 \\
4 & 0 & 1 \\
1 & 2 & 2
\end{bmatrix}
$$

$$
\det(A) = \det\begin{bmatrix}
2 & 1 & 3 \\
4 & 0 & 1 \\
1 & 2 & 2
\end{bmatrix}
$$

#### Row Operations to Create Upper Triangular Form

**Operation 1**: $R_2 \leftarrow R_2 - 2R_1$ (no change to determinant)

$$
= \det\begin{bmatrix}
2 & 1 & 3 \\
0 & -2 & -5 \\
1 & 2 & 2
\end{bmatrix}
$$

**Operation 2**: $R_3 \leftarrow R_3 - \frac{1}{2}R_1$ (no change to determinant)

$$
= \det\begin{bmatrix}
2 & 1 & 3 \\
0 & -2 & -5 \\
0 & \frac{3}{2} & \frac{1}{2}
\end{bmatrix}
$$

**Operation 3**: $R_3 \leftarrow R_3 + \frac{3}{4}R_2$ (no change to determinant)

$$
= \det\begin{bmatrix}
2 & 1 & 3 \\
0 & -2 & -5 \\
0 & 0 & -\frac{13}{4}
\end{bmatrix}
$$

For an upper triangular matrix, the determinant is the product of diagonal elements:

$$\det(A) = 2 \cdot (-2) \cdot \left(-\frac{13}{4}\right) = \frac{52}{4} = 13$$

---

### Elementary Matrices

An **elementary matrix** is a square matrix obtained from the identity matrix by performing exactly one elementary row operation. Pre-multiplying any matrix by an elementary matrix applies that same row operation to the matrix as a whole.

-   If $E$ is elementary and $D$ is any $m\times n$ matrix, then $E\,D$ is $D$ with one row operation applied.
-   Right-multiplication by an elementary matrix applies the corresponding column operation: $D\,E$.

---

## Laplace (Cofactor) Expansion for Determinants

For an $n \times n$ matrix $A = [a_{ij}]$, the Laplace expansion along row $i$ is:

$$\det(A) = \sum_{j=1}^n (-1)^{i+j} a_{ij} \det(M_{ij})$$

where:

-   $M_{ij}$ is the $(n-1) \times (n-1)$ **minor** obtained by deleting row $i$ and column $j$
-   $C_{ij} = (-1)^{i+j}\det(M_{ij})$ is called the **cofactor**

You can expand along any row or any column — pick one with zeros to simplify.

## Step-by-Step Example

Let's compute the determinant of:

$$
A = \begin{bmatrix}
2 & 0 & 3 \\
1 & 4 & 5 \\
0 & 2 & 1
\end{bmatrix}
$$

Row 3 has a zero in position (3,1), so let's expand along row 3 to minimize calculations.

#### Expansion Along Row 3

$$\det(A) = \sum_{j=1}^3 (-1)^{3+j} a_{3j} \det(M_{3j})$$

Breaking this down:

-   $j=1$: $(-1)^{3+1} \cdot 0 \cdot \det(M_{31}) = 0$
-   $j=2$: $(-1)^{3+2} \cdot 2 \cdot \det(M_{32})$
-   $j=3$: $(-1)^{3+3} \cdot 1 \cdot \det(M_{33})$

#### Computing the Minors

**For $M_{32}$** (delete row 3, column 2):

$$
M_{32} = \begin{bmatrix}
2 & 3 \\
1 & 5
\end{bmatrix}
$$

$$\det(M_{32}) = 2(5) - 3(1) = 10 - 3 = 7$$

**For $M_{33}$** (delete row 3, column 3):

$$
M_{33} = \begin{bmatrix}
2 & 0 \\
1 & 4
\end{bmatrix}
$$

$$\det(M_{33}) = 2(4) - 0(1) = 8$$

### Final Calculation

$$\det(A) = (-1)^{3+2} \cdot 2 \cdot 7 + (-1)^{3+3} \cdot 1 \cdot 8$$
$$= (-1)^5 \cdot 2 \cdot 7 + (-1)^6 \cdot 1 \cdot 8$$
$$= -14 + 8$$
$$= -6$$

#### Alternative: Expansion Along Column 2

Since column 2 has a zero at position (1,2), we could also expand along column 2:

$$\det(A) = \sum_{i=1}^3 (-1)^{i+2} a_{i2} \det(M_{i2})$$

Only non-zero terms:

-   $i=2$: $(-1)^{2+2} \cdot 4 \cdot \det(M_{22}) = 4 \cdot 2 = 8$
-   $i=3$: $(-1)^{3+2} \cdot 2 \cdot \det(M_{32}) = -2 \cdot 7 = -14$

Result: $8 + (-14) = -6$

#### Key Points

1. **Choice matters**: Pick rows/columns with the most zeros
2. **Sign pattern**: $(-1)^{i+j}$ creates a checkerboard pattern:
    $$
    \begin{bmatrix}
    + & - & + & \cdots \\
    - & + & - & \cdots \\
    + & - & + & \cdots \\
    \vdots & \vdots & \vdots & \ddots
    \end{bmatrix}
    $$

---

### Property of Determinants

1. **Transpose**: $\det(A^T) = \det(A)$

2. **Inverse**: $\det(A^{-1}) = \frac{1}{\det(A)}$ (when $A$ is invertible)

3. **Scalar multiplication**: $\det(kA) = k^n \det(A)$ for $n \times n$ matrix

4. **Identity**: $\det(I) = 1$

5. **Triangular matrices**: Determinant equals product of diagonal entries

$$
  \det(A) = \prod_{i=1}^n a_{ii}.
$$

6. **Product rule**: $\det(AB) = \det(A) \cdot \det(B)$

    - **Matrix multiplication** itself is NOT commutative. In general, $AB \neq BA$.
    - The **determinant** of a matrix is a single number (a scalar). Scalar multiplication IS commutative.

---

### The Gauß Algorithm (Gaussian Elimination)

The **Gauß algorithm** (Gaussian elimination) is a systematic method for solving systems of linear equations by transforming the coefficient matrix into **upper triangular form** using elementary row operations.

An upper triangular matrix has all zeros below the main diagonal:

$$
\begin{bmatrix}
* & * & * & * \\
0 & * & * & * \\
0 & 0 & * & * \\
0 & 0 & 0 & *
\end{bmatrix}
$$

where $*$ represents any value (including zero).

1. **Choose pivot**: Select the leftmost non-zero entry in the first row
2. **Eliminate below**: Use row operations to create zeros below the pivot
3. **Move to next row/column**: Repeat for subsequent rows
4. **Continue** until upper triangular form is achieved

#### Example

Transform this matrix to upper triangular form:

$$
A = \begin{bmatrix}
2 & 1 & -1 \\
-3 & -1 & 2 \\
-2 & 1 & 2
\end{bmatrix}
$$

#### Step 1: First column

Eliminate below pivot (2):

$R_2 \leftarrow R_2 + \frac{3}{2}R_1$:

$$
\begin{bmatrix}
2 & 1 & -1 \\
0 & \frac{1}{2} & \frac{1}{2} \\
-2 & 1 & 2
\end{bmatrix}
$$

$R_3 \leftarrow R_3 + R_1$:

$$
\begin{bmatrix}
2 & 1 & -1 \\
0 & \frac{1}{2} & \frac{1}{2} \\
0 & 2 & 1
\end{bmatrix}
$$

#### Step 2: Second column

Eliminate below second pivot ($\frac{1}{2}$):

$R_3 \leftarrow R_3 - 4R_2$:

$$
\begin{bmatrix}
2 & 1 & -1 \\
0 & \frac{1}{2} & \frac{1}{2} \\
0 & 0 & -1
\end{bmatrix}
$$

Once in upper triangular form, we can **calculate Determinant** by:

$$\det(A) = \pm \prod_{i=1}^n a_{ii}$$
(product of diagonal entries, with sign depending on number of row swaps)

---

## Inverse of a 2×2 Matrix

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

#### Minor Matrix

For an $n \times n$ square matrix $A = (a_{ij})$, the **minor** $M_{ij}$ of entry $a_{ij}$ is the determinant of the $(n-1) \times (n-1)$ submatrix obtained by deleting the $i$-th row and $j$-th column of $A$.

For $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

The minor $M_{12}$ (delete row 1, column 2):
$$M_{12} = \det\begin{bmatrix} 4 & 6 \\ 7 & 9 \end{bmatrix} = 36 - 42 = -6$$

The **cofactor** of $a_{ij}$ is defined as:

$$C_{ij} = (-1)^{i+j} M_{ij}$$

The factor $(-1)^{i+j}$ creates a "checkerboard" pattern of signs:

$$
\begin{bmatrix}
+ & - & + & \cdots \\
- & + & - & \cdots \\
+ & - & + & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

$$C_{12} = (-1)^{1+2} M_{12} = (-1)^3 \cdot (-6) = -1 \cdot (-6) = 6$$

The **cofactor matrix** of $A$ is the matrix whose entries are the cofactors:

$$
\text{Cof}(A) = \begin{bmatrix}
C_{11} & C_{12} & \cdots & C_{1n}\\
C_{21} & C_{22} & \cdots & C_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
C_{n1} & C_{n2} & \cdots & C_{nn}
\end{bmatrix}
$$

The **adjugate matrix** (or adjoint) is the transpose of the cofactor matrix:

$$\text{adj}(A) = \text{Cof}(A)^T$$

#### Inverse Formula

When $A$ is invertible:

$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

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

$$
A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}
$$

which is already in RREF.

#### 1. Identify Pivot and Non-Pivot Columns

Pivots appear at the first nonzero entry of each nonzero row:

-   Row 1 pivot → column 1
-   Row 2 pivot → column 3
-   Row 3 pivot → column 4

Therefore:

-   **Pivot columns:** 1, 3, 4
-   **Non-pivot columns:** 2, 5

#### 2. Express Non-Pivot Columns as Linear Combinations of Pivot Columns

Write the column vectors (columns of $A$) explicitly:

$$
\begin{aligned}
p_1 &= \text{col}_1(A) = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, &
p_2 &= \text{col}_2(A) = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}, \\
p_3 &= \text{col}_3(A) = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, &
p_4 &= \text{col}_4(A) = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \\
p_5 &= \text{col}_5(A) = \begin{bmatrix} 3 \\ 9 \\ -4 \end{bmatrix}.
\end{aligned}
$$

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

$$
\begin{cases}
x_1 + 3x_2 + 3x_5 = 0 \\
x_3 + 9x_5 = 0 \\
x_4 - 4x_5 = 0
\end{cases}
$$

#### 4. Parameterize the Solution (Free Variables)

Non-pivot columns correspond to free variables. Let:

$$x_2 = s, \quad x_5 = t \quad (s, t \in \mathbb{R})$$

Then from the three equations above:

$$
\begin{aligned}
x_1 &= -3x_2 - 3x_5 = -3s - 3t \\
x_3 &= -9x_5 = -9t \\
x_4 &= 4x_5 = 4t
\end{aligned}
$$

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

$$
A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}
$$

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

$$
\begin{cases}
x_1 + 3x_2 + 3x_5 = 0 \\
x_3 + 9x_5 = 0 \\
x_4 - 4x_5 = 0
\end{cases}
$$

### Geometrically

If we think of $A$ as a linear transformation from $\mathbb{R}^5 \to \mathbb{R}^3$, the null space is the set of all vectors in $\mathbb{R}^5$ that are mapped to the zero vector in $\mathbb{R}^3$. In this case, $\ker(A)$ is a two-dimensional subspace of $\mathbb{R}^5$ (a "plane" in 5-dimensional space) spanned by the basis vectors above.

### Dimension of the Null Space

The dimension of $\ker(A)$, called the **nullity**, equals the number of free variables (non-pivot columns). Here:

-   Total columns: 5
-   Pivot columns: 3
-   Non-pivot columns: $5 - 3 = 2$
-   Therefore, nullity = 2

## Importance of the Null Space

### 1. Solutions to Homogeneous Systems

-   If $\ker(A) = \{ \mathbf{0} \}$: only the trivial solution exists
-   If $\ker(A)$ contains non-zero vectors: infinitely many solutions exist

### 2. Linear Independence

-   If $\ker(A) = \{ \mathbf{0} \}$: columns of $A$ are linearly independent
-   If $\ker(A)$ is non-trivial: columns of $A$ are linearly dependent

### 3. Rank-Nullity Theorem

$$\text{rank}(A) + \text{nullity}(A) = n$$

For our example:

-   $\text{rank}(A) = 3$ (number of pivot columns)
-   $\text{nullity}(A) = 2$ (dimension of null space)
-   $n = 5$ (number of columns)

---

### Mechanical Method for Constructing Null Space Basis from RREF : The Free‑Variable Identity‑Block Pattern

This method leverages the structure of the RREF to systematically find the solutions to the homogeneous system $A\mathbf{x} = \mathbf{0}$.

#### Short Algorithm (Mechanical Recipe)

### Step-by-Step Method

1. **Identify pivot and free columns**

    - Pivot columns: those with a leading 1 in each non-zero row of RREF
    - Free columns: those without pivots (corresponding to free variables)
    - Let $A \in \mathbb{R}^{m\times n}$ be in RREF with
      pivot columns indexed by $I = (i_1,\dots,i_r)$,
      free columns indexed by $J = (j_1,\dots,j_k)$, where $k = n - r$.

2. **Construct basis vectors**
   For each free variable:

    - Set that free variable to 1 and all other free variables to 0
    - For each pivot variable, set its value to the negative of the coefficient of the free variable in that pivot row
    - The resulting vector is a basis vector for the null space

3. **Form the basis matrix**
    - Collect coefficients of free variables in pivot rows into matrix $C$
    - Construct null space basis matrix $N = \begin{bmatrix} -C \\ I_k \end{bmatrix}$

-   Each pivot row has the form
    $$
    x_{i_p} + \sum_{q=1}^k C_{p,q}\,x_{j_q} = 0,\qquad p=1,\dots,r,
    $$
    which defines the coefficient matrix $C \in \mathbb{R}^{r\times k}$ of the free variables in the pivot equations.

For an $m \times n$ matrix in RREF with $r$ pivot columns and $k = n - r$ free columns:

### Setup

-   Pivot indices: $i_1, i_2, \ldots, i_r$ (pivot variables)
-   Free indices: $j_1, j_2, \ldots, j_k$ (free variables)

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

---

Consider the matrix:

$$
A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}
$$

This is a $3 \times 5$ matrix in RREF. We seek all $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5]^T$ such that $A\mathbf{x} = \mathbf{0}$.

##### Step 1: Identify Pivot and Free Columns

**Pivot columns:**

-   Row 1, column 1 ($x_1$)
-   Row 2, column 3 ($x_3$)
-   Row 3, column 4 ($x_4$)

Therefore, pivot columns are 1, 3, and 4, corresponding to variables $x_1, x_3, x_4$.

**Free columns:** Columns 2 and 5 have no pivots, corresponding to free variables $x_2$ and $x_5$. There are $k = 2$ free variables.

##### Step 2: Read Coefficients of Free Variables

The pivot rows give equations involving free variables $x_2$ and $x_5$:

-   Row 1: $x_1 + 3x_2 + 3x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(3, 3)$
-   Row 2: $x_3 + 9x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(0, 9)$
-   Row 3: $x_4 - 4x_5 = 0$ → Coefficients of $(x_2, x_5)$ are $(0, -4)$

Collect these into matrix $C$:

$$
C = \begin{bmatrix}
3 & 3 \\
0 & 9 \\
0 & -4
\end{bmatrix}
$$

### Step 3: Construct Null Space Basis Matrix $N$

The null space basis matrix is:

$$
N = \begin{bmatrix} -C \\ I_2 \end{bmatrix} = \begin{bmatrix}
-3 & -3 \\
0 & -9 \\
0 & 4 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$

Reordering rows to match variable order $(x_1, x_2, x_3, x_4, x_5)$:

$$
N = \begin{bmatrix}
-3 & -3 \\
1 & 0 \\
0 & -9 \\
0 & 4 \\
0 & 1
\end{bmatrix}
$$

### Result: Basis Vectors

The columns of $N$ are the basis vectors:

$$v^{(1)} = \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \quad v^{(2)} = \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix}$$

These correspond to:

-   For $x_2$: Set $x_2 = 1$, $x_5 = 0$, then $x_1 = -3$, $x_3 = 0$, $x_4 = 0$
-   For $x_5$: Set $x_2 = 0$, $x_5 = 1$, then $x_1 = -3$, $x_3 = -9$, $x_4 = 4$

The null space is:

$$\ker(A) = \text{span} \left\{ \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ -9 \\ 4 \\ 1 \end{bmatrix} \right\}$$

---

### The Minus-1 Trick for Homogeneous Systems

The Minus-1 Trick is a practical method for reading out solutions $\mathbf{x}$ of a homogeneous system of linear equations $A\mathbf{x} = \mathbf{0}$, where $A \in \mathbb{R}^{k \times n}$, $\mathbf{x} \in \mathbb{R}^n$.

We assume that $A$ is in reduced row-echelon form without any rows that contain only zeros

#### The Augmentation Process

1. **Extend the matrix**: We extend this $k \times n$ matrix $A$ to an $n \times n$ matrix $\tilde{A}$ by adding $n - k$ rows

2. **Form of added rows**: Each added row has the form:
   $$[0 \cdots 0 \quad -1 \quad 0 \cdots 0] $$
   where $-1$ is placed in a non-pivot column position

3. **Result**: The diagonal of the augmented matrix $\tilde{A}$ contains either 1 or $-1$

#### Key Result

**The columns of $\tilde{A}$ that contain $-1$ as pivots are solutions of the homogeneous system $A\mathbf{x} = \mathbf{0}$.**

## Example

Given

$$
A=\begin{bmatrix}
1 & 3 & 0 & 0 & 3\\
0 & 0 & 1 & 0 & 9\\
0 & 0 & 0 & 1 & -4
\end{bmatrix}
\quad (k=3,\ n=5),
$$

We now augment this matrix to a 5 × 5 matrix by adding rows of the form [ 0 · · · 0 −1 0 · · · 0 ] at the places where the pivots on the diagonal are missing and obtain

$$
\tilde A=
\begin{bmatrix}
1 & 3 & 0 & 0 & 3\\
0 & -1 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 9\\
0 & 0 & 0 & 1 & -4\\
0 & 0 & 0 & 0 & -1
\end{bmatrix}.
$$

From the augmented matrix $\tilde{A}$, we can immediately read out the solutions of $A\mathbf{x} = \mathbf{0}$ by taking the columns of $\tilde{A}$ which contain $-1$ on the diagonal.

For $\mathbf{x} \in \mathbb{R}^5$:

$$\mathbf{x} = \lambda_1 \begin{bmatrix} 3 \\ -1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\ 0 \\ 9 \\ -4 \\ -1 \end{bmatrix}, \quad \lambda_1, \lambda_2 \in \mathbb{R}$$

---

### Comparison: Standard Method vs. Minus-1 Trick

`Standard "Identity Block" Form`

If (after permuting columns so pivots come first) we have:

$$A = [I_k \; C]$$

then the system $A\mathbf{x} = \mathbf{0}$ becomes:

$$I_k \mathbf{x}_p + C\mathbf{x}_f = \mathbf{0} \quad \implies \quad \mathbf{x}_p = -C\mathbf{x}_f$$

where:

-   $\mathbf{x}_p \in \mathbb{R}^k$ are the pivot variables
-   $\mathbf{x}_f \in \mathbb{R}^{n-k}$ are the free variables

Thus the null space basis is packaged in the matrix:

$$N = \begin{bmatrix} -C \\ I_{n-k} \end{bmatrix}$$

`Minus-1 Trick Form`

When you extend $A$ to the augmented $\tilde{A}$ with "$-1$ pivots" for free variables, the free-variable columns in $\tilde{A}$ look like:

$$\begin{bmatrix} C \\ -I_{n-k} \end{bmatrix}$$

So the Minus-1 trick produces:

$$\tilde{N} = \begin{bmatrix} C \\ -I_{n-k} \end{bmatrix}$$

Since multiplying a basis vector by $-1$ does not change the span, both methods describe exactly the same null space $\ker(A)$:

$$\text{span}(N) = \text{span}(\tilde{N}) = \ker(A)$$

---

## Calculating the Inverse Using Gaussian Elimination

To compute the inverse $A^{-1}$ of $A \in \mathbb{R}^{n \times n}$, we need to find a matrix $X$ that satisfies $AX = I_n$. Then, $X = A^{-1}$.

We can write this as a set of simultaneous linear equations $AX = I_n$, where we solve for $X = [\mathbf{x}_1 | \cdots | \mathbf{x}_n]$.

Using augmented matrix notation for a compact representation:

$$[A | I_n] \leadsto \cdots \leadsto [I_n | A^{-1}] $$

This means that if we bring the augmented equation system into reduced row-echelon form, we can read out the inverse on the right-hand side of the equation system.

If the left block cannot be reduced to $I_n$ (i.e., a zero row appears), then $A$ is singular and $A^{-1}$ does not exist.

**Key insight**: Determining the inverse of a matrix is equivalent to solving systems of linear equations.

#### Example

To determine the inverse of:

$$
A = \begin{bmatrix}
1 & 0 & 2 & 0 \\
1 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

#### Step 1: Set up the augmented matrix

$$
\left[\begin{array}{cccc|cccc}
1 & 0 & 2 & 0 & 1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 & 0 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 1
\end{array}\right]
$$

#### Step 2: Apply Gaussian elimination to reach RREF

$$
\left[\begin{array}{cccc|cccc}
1 & 0 & 0 & 0 & -1 & 2 & -2 & 2 \\
0 & 1 & 0 & 0 & 1 & -1 & 2 & -2 \\
0 & 0 & 1 & 0 & 1 & -1 & 1 & -1 \\
0 & 0 & 0 & 1 & -1 & 0 & -1 & 2
\end{array}\right]
$$

#### Step 3: Read the inverse from the right-hand side

The desired inverse is given as:

$$
A^{-1} = \begin{bmatrix}
-1 & 2 & -2 & 2 \\
1 & -1 & 2 & -2 \\
1 & -1 & 1 & -1 \\
-1 & 0 & -1 & 2
\end{bmatrix}
$$

---

## Matrix Inverse and Solving Linear Systems

For a matrix $A$, the inverse $A^{-1}$ is defined so that:

$$AA^{-1} = A^{-1}A = I_n$$

where $I_n$ is the identity matrix of size $n \times n$.

**$A$ must be square ($n \times n$)**, because only then do the dimensions match on both sides for both multiplications.

**If $A$ is square ($n \times n$) and invertible** (i.e., $\det(A) \neq 0$), then:

$$A\mathbf{x} = \mathbf{b} \quad \implies \quad \mathbf{x} = A^{-1}\mathbf{b}$$

This is the clean formula solution for solving linear systems.

#### Non-Invertible Square Matrices

If $A$ is **not invertible** (determinant = 0, or rows/columns are linearly dependent), then $A^{-1}$ does not exist. In that case:

-   The system may have **no solution** (inconsistent system), or
-   The system may have **infinitely many solutions** (dependent system)

#### Non-Square Matrices

If $A$ is **not square** ($m \times n$ with $m \neq n$), Then the usual inverse does not make sense, because $I_m \neq I_n$.

### Generalized Inverses

For non-square or singular matrices, mathematicians define _generalized inverses_.

### (a) Left Inverse

If $A$ is **tall** ($m \geq n$) and has **full column rank (The columns are linearly independent)** (rank = $n$), then:

$$A^{\top}A \in \mathbb{R}^{n \times n}$$

is invertible. We define the **left inverse**:

$$A_L^{-1} = (A^{\top}A)^{-1}A^{\top}$$

which satisfies:

$$A_L^{-1}A = I_n$$

So it behaves like an inverse **on the left**.

#### Why This Formula?

**Requirement**: A left inverse $B$ of $A$ must satisfy $BA = I_n$.

**Construction**:

1. $A^{\top}A$ is always an $n \times n$ symmetric matrix
2. If $A$ has full column rank, then $A^{\top}A$ is invertible
3. Let's verify: $BA = (A^{\top}A)^{-1}A^{\top}A = (A^{\top}A)^{-1}(A^{\top}A) = I_n$ ✓

**Intuition**:

-   The factor $A^{\top}$ projects things back into the column space of $A$
-   Multiplying by $(A^{\top}A)^{-1}$ rescales it correctly so the final effect is identity on $\mathbb{R}^n$
-   This construction comes directly from the **normal equations** in least squares:
    $$A^{\top}A\mathbf{x} = A^{\top}\mathbf{b}$$
    which are solved by:
    $$\mathbf{x} = (A^{\top}A)^{-1}A^{\top}\mathbf{b}$$

### Number of Left Inverses for a Matrix

A left inverse of $A\in\mathbb{R}^{m\times n}$ is a matrix $B\in\mathbb{R}^{n\times m}$ such that

$$
BA = I_n.
$$

The number of left inverses a matrix has depends entirely on its columns. For an $m \times n$ matrix $A$ where $m > n$, there are two possible scenarios:

#### Case 1: Columns are Linearly Independent

If the matrix has **full column rank** (all columns are linearly independent), then it has **infinitely many** left inverses.

### The Pseudoinverse

One of these is the most important and is called the **pseudoinverse**:

$$A_{\text{left}}^{-1} = (A^T A)^{-1} A^T$$

#### General Form

You can create all other left inverses by adding any matrix $N$ that satisfies $NA = 0$:

$$B = A_{\text{left}}^{-1} + N$$

where $NA = 0$ and $B$ is still a left inverse.

#### Example

Consider this $2 \times 1$ matrix (its single column is inherently independent):

$$A = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

**One left inverse**: $B = \begin{bmatrix} 1 & 0 \end{bmatrix}$

Check: $BA = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = [1] = I_1$

**Another left inverse**: $C = \begin{bmatrix} 1 & 5 \end{bmatrix}$

Check: $CA = \begin{bmatrix} 1 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = [1] = I_1$

You could replace 5 with any real number, giving infinitely many left inverses.

#### Case 2: Columns are Linearly Dependent

If the columns are **linearly dependent** (not full column rank), then **no left inverse exists**.

$$
\text{rank}(BA) \le \min\{\text{rank}(B), \text{rank}(A)\} \le \text{rank}(A) < n
$$

**Sylvester's rank inequality**: For compatible matrices,

$$\text{rank}(A) + \text{rank}(B) - n \leq \text{rank}(AB) \leq \min\{\text{rank}(A), \text{rank}(B)\}$$

### (b) Right Inverse

If $A$ is **wide** ($m \leq n$) and has **full row rank** (rank = $m$), then:

$$AA^{\top} \in \mathbb{R}^{m \times m}$$

is invertible. We define the **right inverse**:

$$A_R^{-1} = A^{\top}(AA^{\top})^{-1}$$

-   Check:
    $$
    A A_R^{-1} \;=\; A A^\top (A A^\top)^{-1} \;=\; I_m.
    $$
    Thus $A_R^{-1}$ “undoes” multiplication by $A$ from the right.

### (c) Moore-Penrose Pseudoinverse

### Least Squares Method (overdetermined case)

Often in applications (statistics, machine learning, data fitting), $A$ is **tall** ($m > n$), meaning we have **more equations than unknowns**. In that case, the system is usually **inconsistent** — there is no exact $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{b}$.

Since we cannot solve $A\mathbf{x} = \mathbf{b}$ exactly, we instead look for an $\mathbf{x}$ that makes $A\mathbf{x}$ **as close as possible** to $\mathbf{b}$.

We cannot satisfy all equations, but we can find a vector $x $ that makes the residual

$$
r = b - Ax
$$

as small as possible in the Euclidean norm.

So we solve:

$$
x^\star = \arg \min_x \|Ax - b\|_2.
$$

Formally:

$$\min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|^2$$

This is the **least squares problem**: we want the $\mathbf{x}$ that minimizes the squared error between $A\mathbf{x}$ and $\mathbf{b}$.

#### Derivation (normal equations) :

The error is:

$$f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2 = (A\mathbf{x} - \mathbf{b})^{\top}(A\mathbf{x} - \mathbf{b})$$

Then

$$
\nabla f(x) = 2A^\top(Ax-b).
$$

Setting $\nabla f(x)=0$ gives the **normal equations**

$$
A^\top A\,x = A^\top b.
$$

### Solution

-   If $A$ has full column rank ($\operatorname{rank}(A)=n$), then $A^\top A$ is invertible and the unique minimizer is

    $$
    x^\star = (A^\top A)^{-1}A^\top b.
    $$

    Equivalently, $x^\star = A^+ b$ (pseudoinverse).

-   This is called the least-squares solution because it minimizes the squared error between Ax and 𝑏
-   $A\mathbf{x}$ is the **projection of $\mathbf{b}$** onto the column space of $A$

---

### Minimum-norm solution (underdetermined case)

Underdetermined (wide) system: too many possible solutions, so we pick the one that is shortest in length. That’s the minimum-norm solution.

Among all possible solutions, we select the one with the smallest Euclidean norm:

$$
x^\star = \arg \min_{x : Ax = b} \|x\|_2.
$$

If \(A\) has full row rank, the solution can be expressed as:

$$
x^\star = A^\top (A A^\top)^{-1} b = A^+ b,
$$

where $(A^+)$ denotes the **Moore-Penrose pseudoinverse** of \(A\).

This is called the minimum-norm solution because it picks the shortest vector among the infinitely many possible solutions.

---

### Rank-deficient case

For a matrix $A \in \mathbb{R}^{m \times n}$ the **rank** of (A) is defined as the maximum number of linearly independent columns (or rows) of (A).

If ${rank}(A) < \min(m, n)$ we say that \(A\) is **rank-deficient**.

Intuitively, this means that some columns (or rows) are linearly dependent on others. Consequently:

-   Some directions in the solution space are “free,” leading to **infinitely many solutions** in the underdetermined case.
-   Even in overdetermined least-squares problems, the minimizer may **not be unique**, because one can move along **null-space directions** without changing (Ax).

The **pseudoinverse** $A^+$ defined via the singular value decomposition (SVD), always exists and provides:

-   **Overdetermined case:** Multiple least-squares minimizers exist. The pseudoinverse $(A^+)$ selects the one with **minimum norm**.
-   **Underdetermined case:** Infinitely many solutions exist. The pseudoinverse $(A^+)$ selects the **minimum-norm solution**.

In this case, the **Moore–Penrose pseudoinverse** $A^+$ picks out one special solution:

$$
x^\star = A^+ b,
$$

namely the one with **minimum Euclidean norm** among all least-squares solutions.
This is called the **minimum-norm least-squares solution**, because it not only minimizes $\|Ax - b\|_2$
but also selects the solution with the **smallest Euclidean norm** among all possible least-squares solutions.

---

The **Moore-Penrose pseudoinverse** $A^+$ is a generalization that:

-   Always exists, for any $A \in \mathbb{R}^{m \times n}$
-   No matter what shape or rank A has $𝐴^+$ is defined.
-   Reduces to the usual inverse if $A$ is square and invertible

**Formula** (when $A$ has full column rank):

-   Full column rank ($m\ge n$):
    $$
    A^+ = (A^\top A)^{-1} A^\top \quad (\text{equals } A_L^{-1}).
    $$

**Formula** (when $A$ has full row rank):

-   Full row rank ($m\le n$):
    $$
    A^+ = A^\top (A A^\top)^{-1} \quad (\text{equals } A_R^{-1}).
    $$

The pseudoinverse is particularly important in data science, statistics, and machine learning for **least squares regression**.

For reasons of numerical precision it is generally not recommended to compute the inverse or pseudo-inverse.
Forming $A^\top A$ requires a **matrix–matrix multiplication**, which is computationally expensive for large $A$.
Furthermore, computing the inverse $(A^\top A)^{-1} $ adds additional cost.

### How is it Computed?

The most robust way is via the **Singular Value Decomposition (SVD)**:

$$A = U\Sigma V^{\top}$$

where $U, V$ are orthogonal and $\Sigma$ is diagonal with nonnegative entries (the singular values).

Then:

$$A^+ = V\Sigma^+ U^{\top}$$

where $\Sigma^+$ is obtained by inverting each nonzero singular value and transposing the shape of $\Sigma$.

This construction guarantees:

-   Always exists
-   Stable and numerically reliable
-   Encodes rank and null space information cleanly

`The pseudoinverse` $A^+$ `is a matrix you can always use in place of an inverse.`

---

Geometric view:

$$
Ax^\star = AA^+ b = \operatorname{proj}_{\operatorname{Col}(A)}(b),\qquad
b - Ax^\star \perp \operatorname{Col}(A).
$$

Define the projectors

$$
P_{\operatorname{col}} := AA^+,\qquad P_{\operatorname{row}} := A^+A.
$$

Then $P_{\operatorname{col}}$ projects onto $\operatorname{Col}(A)$ and $P_{\operatorname{row}}$ projects onto $\operatorname{Row}(A)$, both orthogonally:

$$
P_{\operatorname{col}}^2=P_{\operatorname{col}}=(P_{\operatorname{col}})^\top,\quad
P_{\operatorname{row}}^2=P_{\operatorname{row}}=(P_{\operatorname{row}})^\top.
$$

For $x^\star=A^+b$, the residual $r^\star=b-Ax^\star$ satisfies $A^\top r^\star=0$.

---

Gaussian elimination is not only for solving $Ax = b$. The same algorithm underpins many core operations in linear algebra:

-   **Determinants:** You can compute $\det(A)$ by reducing $A$ to triangular form and multiplying the pivots.
-   **Independence:** By row reducing a set of vectors, you can check if they are linearly independent.
-   **Inverses:** By augmenting with $I$ and performing row reduction, one obtains $A^{-1}$.
-   **Rank:** The number of pivots (nonzero rows in echelon form) gives $\text{rank}(A)$.
-   **Bases:** From the reduced row echelon form (RREF), one can extract bases for the column space, row space, or null space.

---

Gaussian elimination has a computational cost of about:

$$O(n^3)$$

operations for an $n \times n$ system.

-   In the **first column**, you eliminate entries below the pivot. That means you update about $(n-1)$ rows, and each update touches about $(n-1)$ entries of that row.  
    → Roughly $(n-1)\cdot (n-1) \approx n^2$ operations.

-   In the **second column**, the active submatrix is now of size $(n-1) \times (n-1)$. Work there is about $(n-2)^2$.

-   Next, $(n-3)^2$, and so on, until the last few pivots are trivial.

The total cost is:

$$
n^2 + (n-1)^2 + (n-2)^2 + \dots + 1^2.
$$

And recall (or prove) the identity:

$$
1^2 + 2^2 + \dots + n^2 = \frac{n(n+1)(2n+1)}{6}.
$$

Thus, the total operation count is on the order of:

$$
\frac{1}{3}n^3.
$$

-   For **millions of variables**: $O(n^3)$ becomes astronomically expensive in both time and memory

---

### Iterative Methods

Instead, we turn to **iterative methods**, which don't try to solve the whole system at once, but **refine an approximation step by step**.

#### General Idea

$$\mathbf{x}^{(k+1)} = C\mathbf{x}^{(k)} + \mathbf{d}$$

where $C$ and $\mathbf{d}$ are chosen so that the approximation converges to the true solution $\mathbf{x}^*$.

-   At each step, the residual:
    $$\mathbf{r}^{(k)} = \mathbf{b} - A\mathbf{x}^{(k)}$$
    gets smaller

-   The error norm $\|\mathbf{x}^{(k)} - \mathbf{x}^*\|$ shrinks with iterations

### Examples of Iterative Methods

-   **Jacobi method**
-   **Gauss-Seidel method**
-   **Successive over-relaxation (SOR)**
-   **Conjugate gradient (CG)** and other **Krylov subspace methods**

These methods scale much better for large, sparse systems because:

-   They don't require full elimination
-   They exploit sparsity (lots of zeros in $A$)
-   Often, convergence is reached in far fewer than $O(n^3)$ steps

        **Gaussian elimination** = exact, finite, but costly → good for moderate-size problems

        **Iterative methods** = approximate but scalable → essential for large systems in science/engineering (e.g., fluid dynamics, machine learning)

---

# Eigenvectors and Eigenvalues

A matrix $A$ represents a linear transformation. An **eigenvector** of a square matrix $A$ is a non-zero vector $\mathbf{v}$ that, when the matrix $A$ is multiplied by $\mathbf{v}$, the direction of $\mathbf{v}$ is unchanged.

The result is simply the original vector $\mathbf{v}$ scaled by some number $\lambda$.

This number $\lambda$ is the **eigenvalue** corresponding to the eigenvector $\mathbf{v}$.

Eigenvalues are defined only for square matrices $ A \in F\_{n \times n} $

### Mathematical Formulation

The defining equation is:

$$\boxed{A\mathbf{v} = \lambda\mathbf{v}}$$

where:

-   $A$ is an $n \times n$ square matrix
-   $\mathbf{v} \neq \mathbf{0}$ is the eigenvector
-   $\lambda$ is the eigenvalue (can be zero, negative, or complex)

### Geometric Interpretation

When a matrix transforms most vectors, it both **rotates** and **scales** them. But eigenvectors are special:

-   They only get **scaled** (stretched or shrunk)
-   Their **direction** remains the same (or flips if $\lambda < 0$)

## Simple 2×2 Example

Consider:
$$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$$

**Eigenvector 1**: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

Check: $A\mathbf{v}_1 = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix} = 3\begin{bmatrix} 1 \\ 0 \end{bmatrix}$

So $\lambda_1 = 3$ ✓

**Eigenvector 2**: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

Check: $A\mathbf{v}_2 = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ -2 \end{bmatrix} = 2\begin{bmatrix} 1 \\ -1 \end{bmatrix}$

So $\lambda_2 = 2$ ✓

## Finding Eigenvalues and Eigenvectors

### Step 1: Find Eigenvalues

Rearrange $A\mathbf{v} = \lambda\mathbf{v}$ to:
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For non-zero $\mathbf{v}$ to exist, the matrix $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0$$

This gives the **characteristic equation**.

### Step 2: Find Eigenvectors

For each eigenvalue $\lambda_i$, solve:
$$(A - \lambda_i I)\mathbf{v} = \mathbf{0}$$

The solutions (excluding $\mathbf{v} = \mathbf{0}$) are the eigenvectors.

#### **Simplifying Computations**

-   Powers of matrices: $$\boxed{A^k\mathbf{v} = \lambda^k\mathbf{v}}$$
-   Matrix exponentials: $e^{At}\mathbf{v} = e^{\lambda t}\mathbf{v}$

#### Proof (by induction on k)

-   Base cases:
    -   $k=0$: $A^0 v = I v = v = \lambda^0 v$.
    -   $k=1$: $A^1 v = A v = \lambda v = \lambda^1 v$ (by definition).
-   Inductive step:
    Assume $A^k v = \lambda^k v$ for some $k\ge 1$. Then
    $$
    A^{k+1} v = A(A^k v) = A(\lambda^k v) = \lambda^k (A v) = \lambda^k(\lambda v) = \lambda^{k+1} v.
    $$
    Hence, by induction, $A^k v = \lambda^k v$ for all $k\in\mathbb{N}$.

If $A$ is invertible, the identity extends to all integers $k\in\mathbb{Z}$:

$$
A^{-k} v = \lambda^{-k} v,
$$

since $A^{-1}v = \lambda^{-1} v$ and the same induction applies.

#### Conceptual view

An eigenvector $v$ specifies an invariant one-dimensional subspace (an “axis”) of the transformation $A$. Acting by $A$ on that axis is simply scaling by $\lambda$. Reapplying $A$ $k$ times composes the same scaling $k$ times, yielding $\lambda^k$ overall.

When you multiply an eigenvector $\mathbf{v}$ by $A$:

1. It stays perfectly on its axis (direction unchanged)
2. It only gets scaled by factor $\lambda$

So applying $A$ exactly $k$ times means:

-   First application: Scale by $\lambda$
-   Second application: Scale by $\lambda$ again
-   Third application: Scale by $\lambda$ again
-   ...
-   $k$-th application: Scale by $\lambda$ again

Total scaling = $\lambda \times \lambda \times \cdots \times \lambda$ ($k$ times) = $\lambda^k$

-   Polynomial functional calculus: For any polynomial $p$,

    $$
    p(A)v = p(\lambda)\,v.
    $$

    ![alt text](/Images/image.png)

-   Analytic functions (when defined via convergent power series): If $f(z)=\sum_{k\ge 0} a_k z^k$ converges at $\lambda$, then

    $$
    f(A)v = \sum_{k\ge 0} a_k A^k v = \sum_{k\ge 0} a_k \lambda^k v = f(\lambda)\,v.
    $$

    Examples: $e^{tA}v = e^{t\lambda} v$, $(A-\mu I)^{-1}v = (\lambda-\mu)^{-1} v$ (when $\lambda\ne \mu$).

-   Instead of computing $A^{100}$ (which requires 99 matrix multiplications), if we know the eigendecomposition:
    $$A^{100}\mathbf{x} = A^{100}(c_1\mathbf{v}_1 + \cdots + c_n\mathbf{v}_n) = c_1\lambda_1^{100}\mathbf{v}_1 + \cdots + c_n\lambda_n^{100}\mathbf{v}_n$$

#### **Understanding System Behavior**

-   **Stability**: If all $|\lambda_i| < 1$, repeated applications of $A$ shrink vectors
-   **Growth**: If any $|\lambda_i| > 1$, the system exhibits growth
-   **Oscillations**: Complex eigenvalues indicate rotational behavior

#### **Applications**

-   **Principal Component Analysis (PCA)**: Eigenvectors of covariance matrices
-   **Google PageRank**: Dominant eigenvector of web link matrix
-   **Quantum Mechanics**: Energy states are eigenvalues
-   **Vibrations**: Natural frequencies are related to eigenvalues

### Special Cases

#### Symmetric Matrices

For real symmetric matrices:

-   All eigenvalues are real
-   Eigenvectors are orthogonal

#### Diagonal Matrices

Eigenvalues are the diagonal entries, eigenvectors are standard basis vectors

#### Identity Matrix

Every non-zero vector is an eigenvector with eigenvalue 1

### Key Properties

1. **Trace**: $\text{tr}(A) = \sum \lambda_i$
2. **Determinant**: $\det(A) = \prod \lambda_i$
3. **Invertibility**: $A$ is invertible iff all $\lambda_i \neq 0$
4. **Diagonalization**: If $A$ has $n$ independent eigenvectors, then $A = PDP^{-1}$ where $D$ is diagonal

---

