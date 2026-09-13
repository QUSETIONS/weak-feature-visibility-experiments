# ICLR 补实验设计 v2（根据 v1 实测改）

> v1 合成主结果成立，不重跑 E1 成对柱和转基曲线。v2 只改两处失败原因，并重跑这两处。

## v1 留下什么、改什么

| 项 | v1 实测 | 处置 |
|---|---|---|
| E1 同 λ 成对 | axis ind 0.096 vs dense 0.969；label/cov 两边都高 | **保留，正文主图左** |
| E1 转基 | ind vs \(1-\lVert u\rVert_4^4\) r=0.977；label 恒为 1 | **保留，正文主图右** |
| E2 对齐 vs Haar | align 0.100 vs Haar 0.920 | **保留** |
| E2-coef | 4 范数赢（0.978）但领先 \(\ell_1\) 只有 0.031 | **重设计方向族后重跑**。v1 的 \(\theta\) 路径上 4 范数与 \(\ell_1\) 共线，门设错了 |
| E4-A「SAE 继承 PCA 轴盲区」 | 未过：λ=0.55 时 axis 的 ind≈0.08 但 TopK cosine 0.94，照样恢复 | **不再作为正假设**。这是预注册 B，写成正结果 |
| E4 诊断类 | `label_only` 恢复 0.056，`cov_not_ind`/`visible` 恢复 1.00 | **保留并加强**：加大 N、向量化、8 seed，作为 SAE 用途 |

## 重新理解（写进正文，不再跟 v1-A 对着干）

三件不是一回事：

1. **固定基上的对角 nuisance 检测**（Theorem 2）：轴对齐特征被坐标方差吃掉，ind 看不见。v1 E1/E2 已经证明。
2. **自由 SAE**：decoder 自选 atom 基，可以把 atom 放在轴上，所以 **不继承 PCA 轴盲区**。v1 E4 已经证伪 A、支持 B。
3. **诊断协议**：label → cov → ind 的类，仍然可以预测「要不要指望这个稀疏 decoder 看见它」。v1 的类-恢复表支持这一点，但检测 N 太小，类挤在 `label_only`。

v2 的 SAE 图不再画「旋转救 SAE」，改画：

**同一 λ、axis vs dense，四列：label / cov / ind / SAE。只有 ind 有几何鸿沟；SAE 两边都高。**

这同时回答 R1（用途=诊断）和 szZ2（收窄 dark matter）。

## E2-sep（新）：分开 4 范数和 \(\ell_1\)

v1 的 \(u(\theta)=( \cos\theta,\ \sin\theta/\sqrt{m-1},\ldots)\) 让 \(\lVert u\rVert_1\) 和 \(1-\lVert u\rVert_4^4\) 一起涨。v2 用 **2-sparse 不等权**：

\[
u_\varepsilon=(\sqrt{1-\varepsilon},\sqrt{\varepsilon},0,\ldots),\qquad \varepsilon\in\{0.50,0.20,0.05,0.01\}
\]

外加 k-sparse 等权 \(k\in\{1,2,4,8\}\)。

预注册（跑前锁）：

- 主预测子仍是 \(1-\lVert u\rVert_4^4\)：与 ind 功效的 |r| 最大。
- **分离对比**：\(\varepsilon=0.01\)（C≈0，\(\ell_1\approx 1.1\)）ind 功效接近 axis；\(\varepsilon=0.50\)（C=0.5，\(\ell_1=\sqrt{2}\)）ind 明显高于 \(\varepsilon=0.01\)。若功效跟 \(\ell_1\) 走，这两个 2-sparse 不应差这么大。
- 不再要求领先第二名 0.10——那条门被 \(\theta\) 路径污染了。改为：4 范数赢，且分离对比符号正确。

设置：m=8，N=1024，λ=0.55，8 seed，700 trial，与 E1 锁定一致。检测器只有 ind（基=I）。

## E4-v2（重跑）：检测器几何 vs 自由 SAE

数据：居中 Bernoulli，\(p=0.05\)，\(\lambda_{\mathrm{eff}}=a\sqrt{p}\)，m=8。

因子：geometry ∈ {axis, dense} × \(\lambda_{\mathrm{eff}}\in\{0.35,0.55,0.80\}\) × 8 seed。

检测：N=4096，400 trial，α=0.05，Monte Carlo 零。类规则与 v1 相同（功效≥0.5 算过线）。

SAE：TopK，dict=16，k=2，N_train=20000，4000 step，cosine≥0.80 为恢复。阈值冻结，不看结果再改。

预注册：

| 行 | 预言 | 推翻 |
|---|---|---|
| E4-geom-ind | λ=0.55：axis ind ≤0.15，dense ind ≥0.80 | 几何鸿沟消失 |
| E4-geom-SAE | λ=0.55：axis 与 dense 的 SAE 恢复率差 \|Δ\| ≤ 0.25 | 若 axis 恢复率仍低 0.25 以上，则回到 v1-A（代理紧） |
| E4-class | 恢复率 `label_only` < `cov_not_ind` ≤ `visible`（允许并列 1） | 类无序或反序 |
| E4-not-A | 不声称 SAE 继承 PCA 轴盲区 | — |

正文 Fig.ICLR-2：λ=0.55 的四列柱（label/cov/ind/SAE × axis/dense）。附录：按类的恢复率。

## 明确仍不做

GPT-2 桥、自然事件、whitener 网格：A100 上没有旧 cache，v2 不编这些数。E1 主图不重跑。不和其他 SAE 架构 bake-off。
