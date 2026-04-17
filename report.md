**Stage 2 — L1稀疏多变量筛选**

对回归任务使用 **Elastic Net**（弹性网络正则化）：

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\lVert\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\rVert_2^2 + \lambda\left[\alpha\lVert\boldsymbol{\beta}\rVert_1 + \frac{1-\alpha}{2}\lVert\boldsymbol{\beta}\rVert_2^2\right] \right\}
$$

其中 $\lambda$ 为正则化强度，$\alpha\in[0,1]$ 控制L1与L2混合比（通过交叉验证自动选择 $\alpha^* = 0.10$，$\lambda^* = 15.06$）。

对分类任务使用 **L1正则化Logistic回归**（Lasso Logistic）：

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ -\sum_{i=1}^n \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right] + \lambda\lVert\boldsymbol{\beta}\rVert_1 \right\}
$$

$$
p_i = \sigma(\mathbf{x}_i^\top\boldsymbol{\beta}) = \frac{1}{1+e^{-\mathbf{x}_i^\top\boldsymbol{\beta}}}
$$

L1惩罚将不重要特征的系数**压缩为零**，实现自动稀疏特征选择。取固定 $C=0.1$（$C=1/\lambda$）。