# 作业 1：正交设计下 Ridge 与 OLS 的关系

设线性回归模型为 $y=X\beta+\varepsilon$，Ridge 回归的目标函数为

$$
\min_\beta \ \|y-X\beta\|_2^2+\lambda\|\beta\|_2^2,\qquad \lambda\geq0.
$$

其闭式解为

$$
\hat\beta^{\mathrm{Ridge}}=(X^\mathsf{T}X+\lambda I)^{-1}X^\mathsf{T}y.
$$

若设计矩阵的列已标准正交，即 $X^\mathsf{T}X=I$，则

$$
\begin{aligned}
\hat\beta^{\mathrm{Ridge}}
&=(I+\lambda I)^{-1}X^\mathsf{T}y\\
&=\frac{1}{1+\lambda}X^\mathsf{T}y.
\end{aligned}
$$

另一方面，OLS 的解为

$$
\hat\beta^{\mathrm{OLS}}=(X^\mathsf{T}X)^{-1}X^\mathsf{T}y=I^{-1}X^\mathsf{T}y=X^\mathsf{T}y.
$$

因此

$$
\boxed{\hat\beta^{\mathrm{Ridge}}=\frac{1}{1+\lambda}\hat\beta^{\mathrm{OLS}}.}
$$

这说明：在正交设计下，Ridge 将每一个 OLS 系数按**完全相同的比例** $1/(1+\lambda)$ 向 0 收缩；$\lambda=0$ 时退化为 OLS，$\lambda$ 增大时收缩更强。

> 题目中的 `β^Ridge=11+λβ^OLS` 应理解为 $\beta^{\mathrm{Ridge}}=\frac{1}{1+\lambda}\beta^{\mathrm{OLS}}$，而非 $11+\lambda$。