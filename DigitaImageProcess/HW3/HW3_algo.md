考虑到图像分割不仅是颜色的聚类，更是空间实体的提取，我决定放弃传统的一维灰度或三维 RGB 模型，构建一个五维特征空间。

为了消除光照梯度对背景分割的影响，我选择在 **CIELAB** 空间进行推导。在该空间中，亮度 $L$ 与色度 $a, b$ 是解耦的。

每个像素 $i$ 的原始特征表示为：
$$
\mathbf{f}_i = [L_i, a_i, b_i, x_i, y_i]^T
$$
由于坐标系 $(x, y)$ 的数值范围（通常为图像的长宽像素）远大于颜色值（$0-100$ 或 $-128-127$），我意识到如果不进行归一化，空间距离将完全主导聚类结果，导致图像被机械地切成方块。

为了解决这个问题，我引入了 **归一化算子** 和 **空间权重系数 $\lambda$**：

1. **颜色归一化**：将 $L, a, b$ 映射到 $[0, 1]$。
   $$
   L'_i = \frac{L_i}{100}, \quad a'_i = \frac{a_i + 128}{255}, \quad b'_i = \frac{b_i + 128}{255}
   $$

2. **空间归一化与加权**：将坐标映射到 $[0, 1]$ 并乘以 $\lambda$。
   $$
   x'_i = \lambda \cdot \frac{x_i}{W}, \quad y'_i = \lambda \cdot \frac{y_i}{H}
   $$

由此，我得到了最终用于推导的归一化 5D 特征向量：
$$
\mathbf{v}_i = [L'_i, a'_i, b'_i, x'_i, y'_i]^T
$$
为了实现最优分割，我将问题转化为一个最小化簇内误差平方和（SSE）的优化问题。我定义目标函数 $J$ 为所有像素点 $\mathbf{v}_i$ 到其所属类中心 $\boldsymbol{\mu}_j$ 的欧几里得距离平方之和：
$$
J = \sum_{j=1}^{K} \sum_{i \in S_j} \| \mathbf{v}_i - \boldsymbol{\mu}_j \|^2
$$
其中，$K$ 是聚类总数 ，$S_j$ 是第 $j$ 个簇的集合。  

由于 $J$ 的变量包括像素分配关系和中心点位置，我采用迭代优化的方式进行求解。在聚类中心 $\boldsymbol{\mu}_j$ 固定的情况下，为了使 $J$ 最小，对于每个像素 $i$，我必须寻找使其距离最小的 $j$。这在数学上体现为求偏导的离散化搜索：
$$
Label_i = \arg \min_{j} \left( \sum_{d=1}^{5} (v_{i,d} - \mu_{j,d})^2 \right)
$$
在像素归属关系 $S_j$ 固定的情况下，我需要求出使 $J$ 达到极小值的中心点 $\boldsymbol{\mu}_j$。我对 $J$ 关于 $\boldsymbol{\mu}_j$ 求偏导：
$$
\frac{\partial J}{\partial \boldsymbol{\mu}_j} = \frac{\partial}{\partial \boldsymbol{\mu}_j} \sum_{i \in S_j} (\mathbf{v}_i - \boldsymbol{\mu}_j)^T (\mathbf{v}_i - \boldsymbol{\mu}_j)
$$
利用矩阵微分规则 $\frac{\partial (x-a)^T(x-a)}{\partial a} = -2(x-a)$，得到：
$$
\frac{\partial J}{\partial \boldsymbol{\mu}_j} = -2 \sum_{i \in S_j} (\mathbf{v}_i - \boldsymbol{\mu}_j)
$$
令偏导数为 0 以求极值点：
$$
-2 \sum_{i \in S_j} (\mathbf{v}_i - \boldsymbol{\mu}_j) = 0
$$

$$
\sum_{i \in S_j} \mathbf{v}_i - \sum_{i \in S_j} \boldsymbol{\mu}_j = 0
$$

$$
\sum_{i \in S_j} \mathbf{v}_i = N_j \boldsymbol{\mu}_j
$$

最终解得：
$$
\boldsymbol{\mu}_j = \frac{1}{N_j} \sum_{i \in S_j} \mathbf{v}_i
$$
***结论**：新的聚类中心必须是当前簇内所有像素在 5D 空间中的算术平均值。*

之后为了实现老师案例中那种直观的“色彩标签”效果 ，我将最终收敛的 $K$ 个聚类标签映射到一个高对比度的离散调色板中。  输出像素颜色 $C_{out}$ 的逻辑如下：
$$
C_{out}(x, y) = \text{Palette}(Label_{x, y})
$$
这一步将复杂的连续图像信号转化为了具有语义意义的离散对象图。

### 总结

通过上述推导就会实现一个具备空间感知能力的分割模型：

- LAB 转换解决了光照梯度带来的分类破碎问题。
- 归一化 5D 特征平衡了色彩语义与空间距离。
- SSE 目标函数优化确保了分割结果在数学上的局部最优性。

