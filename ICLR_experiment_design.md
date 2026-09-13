# ICLR 补实验设计：Detector-Class Visibility（v1）

> 设计哲学：这是理论论文的实验，不是 SAE 方法文。新实验只做三件事：(i) 把已有相关补上区间；(ii) 用**干预**而不是再报一个相关，证明边界是检测器相对的；(iii) 把 SAE 写成诊断后果，并预注册“代理可能失败”。不训新 SOTA SAE，不做下游 LM，不和其他架构 bake-off。
>
> 版面：ICLR 正文 9 页。新实验正文预算 **1 图 + 1 表 + 旧表加 CI**。其余进附录。旧实验全部保留，只改读法。

现有资产（不要重做，只复用/加区间/加一列）：

| 已有 | 结果 | 缺口 |
|---|---|---|
| 合成：指数 / 几何 / 证据塌缩 / floor | 斜率 −3.97 vs −1.88；轴 0.074 vs dense 1.00；r=0.904 / 0.998 | GPT-2 相关无 CI；同 λ 成对故事不够硬 |
| GPT-2 注入 | r=0.872 / 0.905；轴 label/cov/ind = 1.000/0.445/0.050 | 坐标系没被干预；各向异性只审计没对照 |
| 47 自然事件 | 76.6% BH；r=0.672；几何 min 0.9925 | 没把“几何休眠、强度主导”写成结果 |
| C16 TopK / C17 pretrained SAE | recovery AUC 0.987；r=0.539/0.452/0.339 | 几乎没做同 λ×几何交叉；没接诊断协议 |
| C19 白化/独立残差 | 方向方差 1.035；off-diag 3.87% | 没比 C_white vs C_raw |

---

## 0 正文资产（只允许这三件新东西上正文）

| 正文 | 承载 | 实验 |
|---|---|---|
| **Fig. ICLR-1**（双面板，主图） | 左：同 λ、不同几何的三级检测器（label/cov/ind）；右：只旋转基、不改 λ，ind 功率沿 \(1-\lVert B^\top w\rVert_4^4\) 塌缩 | E1 + E2 |
| **Table ICLR-1**（预注册对账） | 每行一条预言：预测 / 实测 / 推翻条件 / 状态 | 全部新实验 |
| **Fig. ICLR-2**（单面板，可附录优先） | 诊断类 → TopK 恢复率：`ind-invisible` 类恢复低，旋转后升 | E4 |
| 旧 Table 1–3 | 所有相关加 bootstrap 95% CI | E0 |
| 附录 | 白化对照、基选择、自然事件方差分解、敏感性、失败行 | E3/E5/E6 |

旧 Fig.1–2（C17 散点、C18 几何直方图）可留附录，或把 C18 直方图并进 Fig.ICLR-1 的 inset。

---

## 1 冻结的共享协议（所有新实验共用，不许各写一套）

### 1.1 三个检测器（已知方向）

观测 \(h^{(1)},\ldots,h^{(N)}\in\mathbb R^m\)。方向 \(w\) 单位。\(\widehat\Sigma\) 为样本协方差。

| 类 | 统计量 | 对应理论 |
|---|---|---|
| **label** | 标签均值：\(\lVert \bar h_{+}-\bar h_{-}\rVert^2\) 或沿 \(w\) 的两点 \(z\) | \(E_{\mathrm{lab}}\asymp N\lambda^2\) |
| **cov** | 固定背景：\(\lVert \widehat\Sigma-\widehat\Sigma_{\mathrm{bg}}\rVert_F\) 或 \(w^\top(\widehat\Sigma-\sigma^2 I)w\)；背景来自 **无注入的同一 cache**，不在备择上 refit | \(E_{\mathrm{cov}}\asymp N\lambda^4\) |
| **ind** | 对角 nuisance：\(\lVert \mathrm{offdiag}(\widehat\Sigma)\rVert_F\)，或等价地 \(\widehat\Sigma-\mathrm{diag}(\widehat\Sigma)\) 的 Frobenius。坐标是下面指定的基 \(B\) | \(E_{\mathrm{ind}}\asymp N(1-\lVert B^\top w\rVert_4^4)\lambda^4\) |

拒绝域一律 **置换/Monte Carlo 零**，\(\alpha=0.05\)，与旧稿一致。功效 = 700 trial 中拒绝比例，除非另写。种子：沿用 `20260502`，新实验用 `20260824 + k`。

**禁止**：在 ind 检测器里用“已知 \(w\) 的 off-diagonal 投影”之外再塞第一阶均值。label 实验才许用标签。

### 1.2 注入（GPT-2 桥）

沿用旧式，居中 Bernoulli：

\[
h' = h + a(z-p)w,\qquad z\sim\mathrm{Bernoulli}(p),\qquad \lambda_{\mathrm{eff}}=a\sqrt{p}.
\]

默认：GPT-2 small、layer 6 residual、WikiText-2、PCA 白化 fit 在 **train 30k**，检测在 **held-out 30k**。新实验不得改这一条，除非跑 E3/E6 敏感性。

### 1.3 基 \(B\)（这是审稿人问的坐标系，必须写死）

默认分析坐标 = **train PCA 白化坐标** \(B_{\mathrm{PCA}}=U D^{-1/2}\)。  
轴对齐 := \(B_{\mathrm{PCA}} w = e_j\)。  
这不是残差流的神经元轴，也不是 SAE decoder 轴。E2 会把 \(B\) 当成实验因子转起来。

### 1.4 统计纪律

- 功效、相关、AUC：seed 或 event 为独立单位，**1000 次 bootstrap 95% CI**。
- 合成几何实验：8 seed，与旧稿同。
- 相关的比较（\(C_{\mathrm{white}}\) vs \(C_{\mathrm{raw}}\)）：**配对** bootstrap。
- 预注册表在跑前锁一版；失败行照登，不准删点。

---

## 2 实验清单：做什么、不做什么

| ID | 问题 | 决策 | 算力 |
|---|---|---|---|
| **E0** | 旧相关有没有区间？ | **必做**，先做 | CPU 分钟–小时（重跑合成则 ~10 min） |
| **E1** | 同 λ 为何一边可见一边不可见？旋转能不能在不改 λ 时翻盘？ | **必做，正文主图** | CPU 小时级 |
| **E2** | 对角 nuisance 定义在哪个基？盲区是否跟着基走？ | **必做，和 E1 共用主图右面板** | CPU 小时级 |
| **E3** | 各向异性下该不该先白化？ | **必做，附录+正文一句** | 复用 cache，CPU |
| **E4** | 诊断协议能不能预测 SAE miss，旋转能不能救？ | **必做，SAE 唯一新实验** | 小 GPU，数小时 |
| **E5** | 自然事件可见性是几何还是强度/频率？ | **必做，复分析，正文一句** | CPU 分钟 |
| **E6** | whitener / cache / layer 敏不敏感？ | **应做，附录** | CPU |
| SAE bake-off / LM 下游 / 未知方向发现定理 | N2Qr 要的 SOTA | **明确不做** | — |

---

## 3 E0  旧数字补区间（P0，先做）

**问题。** szZ2 点名：主相关没有不确定度。

**协议。** 不改生成过程。对下列量做 bootstrap：

| 量 | 旧点估计 | 重抽样单位 |
|---|---|---|
| 合成 unlabeled / labeled 斜率 | −3.97 / −1.88 | 8 seed |
| 合成证据–功效 r | 0.904 | seed |
| 合成 floor r / MAE | 0.998 / 0.006 | 重采样 360-feature bank |
| GPT-2 证据–功效 r | 0.872 | 735 配置对 bootstrap |
| GPT-2 几何–功效 r | 0.905 | 85 方向 |
| 自然事件 r | 0.672 | 47 events |
| C17 三层 r | 0.539 / 0.452 / 0.339 | 40 events |

**预注册。** CI 覆盖旧点估计；宽度不作为成败标准。若某个 r 的 CI 跨 0，该行降为“弱桥”，不得在正文当验证。

**推翻。** 无。这是报告纪律，不是假说。

---

## 4 E1  同 λ 成对 + 旋转干预（R1 Q3 的直接实验）

**问题。** 两个强度相同的特征，为何可以落在临界边界两侧？若可见性真是 \(C_i\lambda^4\) 而不是 \(\lambda\)，则**只旋转方向、不改 \(\lambda,p,a,N\)** 就应能把 miss 变成 hit。

### 4.1 合成（定理成立处，必须先过）

- \(m=8\)，\(h=\varepsilon+\lambda z w\)，\(\varepsilon\sim N(0,I)\)，与旧稿相同。
- 固定 \(\lambda\in\{0.40,0.55,0.70\}\)，\(N\) 取旧几何实验同一档（与 \(p=0.05,a=0.7,N=8192\) 的 GPT-2 档对齐的合成等效：选使 dense 功效 \(\in[0.6,0.9]\)、axis 功效 \(<0.1\) 的 \(N\)，先在 seed 0 定档再锁）。
- 两个方向：\(w_{\mathrm{axis}}=e_1\)（\(C=0\)），\(w_{\mathrm{dense}}=m^{-1/2}1\)（\(C=1-1/m\)）。
- 三个检测器全报。
- 8 seed × 700 trial。

**预注册（合成）。**

| 检测器 | axis | dense | 必须满足 |
|---|---|---|---|
| label | ≈1 | ≈1 | 两者 CI 重叠，都 \(\ge 0.95\) |
| cov | 中等可见 | 中等可见 | 两者都显著 \(> \alpha\)，几何差 \(\lvert\Delta\rvert<0.15\) |
| ind | \(\le 0.10\) | \(\ge 0.80\) | **同 λ 反号**：axis 在边界下，dense 在上 |

旧稿 axis 0.074、dense 1.00，应复现。失败则先查 ind 统计量是否泄漏了对角。

### 4.2 旋转干预（合成，主图右半的干净版）

固定 \(w=e_1\)、固定 \(\lambda,N\)。取正交阵 \(B_\theta\)，使

\[
B_\theta^\top w = (\cos\theta,\sin\theta/\sqrt{m-1},\ldots),\qquad
\theta\in\{0,15^\circ,30^\circ,45^\circ,90^\circ\}.
\]

ind 检测器在 \(B_\theta\) 坐标里 refit 对角。横轴画 \(1-\lVert B_\theta^\top w\rVert_4^4\)，纵轴画 ind 功效。

**预注册。** 功效对 \(1-\lVert B^\top w\rVert_4^4\) 单调；\(\theta=0\) 近 0；\(\theta=90^\circ\) 接近 dense。Pearson \(r\ge 0.9\)（跨 \(\theta\)×seed）。  
**推翻。** 功效不随 \(\theta\) 动，或随 \(\lVert B^\top w\rVert_\infty\) 比随 4 范数更好（见 E2 竞争预测子）。

### 4.3 GPT-2 注入复现（桥，不是定理）

同一对 \(w_{\mathrm{axis}},w_{\mathrm{dense}}\)，放进 **已经白化** 的 layer-6 cache。\(p=0.05,a=0.7,N=8192\)，与旧 detector-stress 一致，便于和 1.000/0.445/0.050 对上。

**预注册（GPT-2）。**

- label：axis 与 dense 都 \(\ge 0.95\)
- cov：axis \(\in[0.30,0.60]\)（旧 0.445），dense 不低于 axis
- ind：axis \(\le 0.10\)（旧 0.050），dense \(\ge 0.35\)（旧 ~0.48）
- 只旋转白化后的坐标（特征本身的 \(\lambda_{\mathrm{eff}}\) 不变）时，ind 功效随 \(1-\lVert B^\top w\rVert_4^4\) 升，label 功效不变

**推翻。** 旋转后 label 也垮（说明干预动了强度，协议坏了）；或 ind 不随几何动（对角代理在真实激活上不成立——若发生，正文必须改成“代理只在合成成立”，不得藏）。

巧妙点：审稿人问的“同样强为什么一边可见”，用**一张图里的一对柱 + 一条旋转曲线**答完。干预比相关强一个等级。

---

## 5 E2  盲区跟着基走（坐标系 / 代理是否唯一）

**问题。** 对角模型定义在哪个坐标？坐标方差 refit 为什么是 SAE-like independence 的代理？  
**主张（可证伪）。** 盲区不是特征的内在属性，是 **nuisance 切空间的属性**。把 ind 的对角从基 \(B\) 改到 \(B'\)，不可见方向从 \(B^{-1}e_j\) 变到 \(B'^{-1}e_j\)。同一条特征可以在一个基里 ind-invisible、在另一个基里 ind-visible，λ 完全不变。

### 5.1 因子

固定一条注入特征 \(w_\star\)（合成用 \(e_1\)；GPT-2 用一条 PC-orthogonal 的 dense 方向，避免和 PCA 轴重合）。  
ind 的对角分别定义在：

| 基 | 含义 | 预测 ind 可见？ |
|---|---|---|
| \(B_{\mathrm{PCA}}\) | 默认白化坐标 | 若 \(w_\star\) 贴 PCA 轴：否；否则：是 |
| \(B_{\mathrm{rand}}\) | Haar 正交 | 是（以高概率 \(1-\lVert B^\top w\rVert_4^4\approx 1-3/m\)） |
| \(B_{\mathrm{align}}\) | 第一列 = \(w_\star\) | **否**（人为制造盲区） |
| \(B_{\mathrm{SAE}}\) | 预训练 SAE 的 top atoms，QR 成正交基（只在 GPT-2 跑） | 若 \(w_\star\) 接近单 atom：偏低；否则高 |

每个基：合成 8 seed；GPT-2 12 个随机 \(B_{\mathrm{rand}}\) + 1 个 \(B_{\mathrm{align}}\)。

### 5.2 竞争预测子（回答“为什么是 4 范数”）

在同一组方向上，把 ind 功效分别回归到：

\[
1-\lVert u\rVert_4^4,\quad
1-\lVert u\rVert_\infty^2,\quad
\mathrm{PR}=\lVert u\rVert_2^2/\lVert u\rVert_4^4,\quad
\lVert u\rVert_1,\quad
\lambda_{\mathrm{eff}},\quad N.
\]

其中 \(u=B^\top w\)。

**预注册。** \(1-\lVert u\rVert_4^4\) 的 \(|r|\) 最大，且比第二名至少高 0.10（配对 bootstrap）。  
**推翻。** 参与比或 \(\infty\) 范数赢——那时正文仍可讲“散布”，但必须承认闭式系数不是被实验挑出来的。

### 5.3 和 SAE 代理的关系（写进讨论，不写成定理）

E2 **不证明** trained SAE = 对角高斯。它只证明：

> 任何“在固定基上 refit 坐标方差、不能制造新相关”的检测器，都有 \(C=1-\lVert B^\top w\rVert_4^4\) 这一盲区。

SAE-like independence 是这个家族的一员：**基由 decoder 选择，而不是由 PCA 选择**。所以 PCA 轴盲区是家族的边界应力测试；真实 SAE 的盲区（若存在）应改在 atom 基上找。E4 测的是后者紧不紧，E2 测的是家族本身。

---

## 6 E3  白化推论：\(C_{\mathrm{white}}\) vs \(C_{\mathrm{raw}}\)（szZ2 各向异性）

**问题。** 主定理能不能覆盖各向异性背景？便宜推论：先 \(\Sigma^{-1/2}\)，再在白化坐标用同一个 \(C\)。

**协议。** 复用 735 个 GPT-2 注入配置，**不再注入新方向**。对每个配置算两个系数，都拿来预测 **同一** ind 功效：

| 系数 | 定义 |
|---|---|
| \(C_{\mathrm{raw}}\) | 在原始残差坐标，\(\sigma_{\mathrm{eff}}^2=\mathrm{tr}(\Sigma)/m\)，\(C=(1-\lVert w\rVert_4^4)/(4\sigma_{\mathrm{eff}}^4)\) |
| \(C_{\mathrm{PCA}}\) | 默认：train PCA 白化后 \(C=1-\lVert U^\top w\rVert_4^4\)（尺度已吃进白化） |
| \(C_{\mathrm{ZCA}}\) | ZCA 白化后的 4 范数系数 |
| \(C_{\mathrm{diag}}\) | 只做逐维标准化，不正交 |

指标：\(\mathrm{corr}(\log(NC\lambda^4),\ \mathrm{power})\)，配对比较。

**预注册。** \(C_{\mathrm{PCA}}\) 和 \(C_{\mathrm{ZCA}}\) 均优于 \(C_{\mathrm{raw}}\)，\(\Delta r \ge 0.05\)；\(C_{\mathrm{diag}}\) 介于 raw 与 PCA 之间。与旧 0.872 对齐的是 \(C_{\mathrm{PCA}}\)。  
**推翻。** \(C_{\mathrm{raw}}\) 不差于白化——那时“局部各向同性约化”比白化推论更贴数据，正文应改口，不要硬写各向异性定理。

C19 的 holdout 方向方差 1.035 作为 **假设检查** 留在附录，不再当主证据。

---

## 7 E4  诊断协议 → SAE 恢复（R1 的用途；唯一的新 SAE 实验）

**问题。** feature recovery 有什么用？怎么帮 SAE？  
**诚实目标。** 不是提高 GLUE。是：给定一个指定方向，三级检测器把它分成诊断类；诊断类预测 TopK 是否恢复；**几何干预**能在不改 λ 的情况下把类从 miss 推到 recover。

### 7.1 诊断类（冻结规则，跑前锁）

对每个配置，held-out \(\alpha=0.05\)：

| 类 | 规则 | 含义 |
|---|---|---|
| `absent` | label 不拒绝 | 方向上几乎没有一阶证据 |
| `label_only` | label 过，cov 不过 | 有标签可见的均值，二阶不够 |
| `cov_not_ind` | cov 过，ind 不过 | 典型盲区：独立背景把对角吃掉 |
| `visible` | ind 过 | 该检测器类看得见 |

GPT-2 注入网格：沿用旧 \(\lambda_{\mathrm{eff}}\) 范围，**强制包含** axis 与 dense 两族，每族至少 4 个 λ × 3 seed。

**预注册（检测器侧）。** axis 族多数落入 `cov_not_ind`；dense 族在中高 λ 落入 `visible`；极弱 λ 落入 `absent`。`label_only` 允许非空，但不应是轴实验的主质量。

### 7.2 TopK SAE（小、受控、已知方向）

不要预训练新大 SAE。沿 C16：

- 数据：同一注入 cache（合成 Gaussian **和** GPT-2 注入各跑一套；GPT-2 套是桥）
- 字典宽度、k、训练步数：**抄 C16 的 18 配置里已经稳定的那一档**，只加几何因子，不加新超参网格
- 新因子：方向 ∈ {axis, 8-sparse, dense} × \(\lambda_{\mathrm{eff}}\) ∈ {低, 边界下, 边界上, 高} × 3 seed
- 恢复（沿 C16 严格阈值，冻结）：decoder cosine、best-atom label AUC、atom-AUC excess **三项同时过线** 才叫 recovered

**预注册（SAE 侧，两条都写，因为代理可能松）。**

- **A（代理紧）：** 同 \(\lambda_{\mathrm{eff}}\) 下，dense 恢复率 > axis，差 \(\ge 0.3\)；诊断类 `visible` 的恢复率 \(\ge\) `cov_not_ind` \(\ge\) `absent`（单调）。旋转 axis→spread 后，原来的 miss 配置恢复率升 \(\ge 0.3\)。
- **B（代理松，允许照登）：** SAE 对 axis 与 dense 恢复率无显著差。则结论改为：局部对角检测器的盲区 **不等于** 训练 SAE 的盲区（SAE 自选 atom 基，切空间更大）。此时 E1–E2 仍成立，Prop 6 必须保持“recover-or-miss 理想化”，C17 只作弱相关桥。

两条预注册都算成功——失败是“结果藏起来”或“把 B 写成 SAE 理论被验证”。

### 7.3 明确不报的东西

- 不报 GPT-2 语言建模 loss、CE、下游探针 GLUE
- 不和 Gated / JumpReLU / TopK 比 reconstruction
- 不声称诊断类能筛选“语义上正确”的 atom（C17 已经用 random dictionary 当对照，够了）

巧妙点：R1 要“用途”，给的是 **可执行的三级诊断 + 一项干预（旋转）**。N2Qr 要 SOTA，用 C17 的 random-dictionary 对照顶住，不再加架构比赛。

---

## 8 E5  自然事件：几何休眠，强度/频率主导（把审稿人的质疑做成结果）

**问题。** 47 条自然方向全在 \(1-\lVert w\rVert_4^4\ge 0.9925\)。那自然可见性是不是根本不是几何，而是强度、频率、\(\sigma_{\mathrm{eff}}\)？

**协议。** 复分析，不改事件定义。对 off-diagonal \(z\) 或 ind 功效做线性模型（event 为行）：

\[
y \sim \underbrace{\log\lambda_{\mathrm{eff}}}_{\text{强度}} + \underbrace{\log p}_{\text{频率}} + \underbrace{\log\sigma_{\mathrm{eff}}}_{\text{背景}} + \underbrace{1-\lVert w\rVert_4^4}_{\text{几何}}
\]

报告：唯一 \(\Delta R^2\)、标准化系数、bootstrap CI。家族（lexical / POS / NER / cluster）做分层。

**预注册。** 几何项唯一 \(\Delta R^2 < 0.05\)，CI 含 0；强度或频率至少一项 CI 不含 0。这与 C18 一致，写成**正结果**：自然 GPT-2 事件处在高几何区，理论的几何旋钮在这里饱和，可见性由 \(\lambda,p,\sigma_{\mathrm{eff}}\) 驱动。轴实验仍是边界应力测试。

**推翻。** 几何项在自然银行里仍有大唯一 \(R^2\)——与 C18 的“远离轴”矛盾，先查方向估计是否塌缩到 PCA 轴。

---

## 9 E6  敏感性（附录，堵住“只审计了一次白化”）

因子（全是检测器，不训 SAE）：

| 因子 | 水平 |
|---|---|
| whitener | PCA / ZCA / diag-std |
| cache | 12k（C19）/ 30k（主实验） |
| layer | 4 / 6 / 8（POS 子集即可，旧稿已有 4/6/8/10） |
| \(\alpha\) | 0.01 / 0.05 |

指标：E1 的 axis/dense ind 功效差；E3 的 \(r(C,\mathrm{power})\)。

**预注册。** 符号不改：axis ind 低、dense 高；\(C_{\mathrm{white}}\) 仍优于 \(C_{\mathrm{raw}}\)。绝对值允许动。  
**推翻。** 任一 whitener 上轴/dense 符号反转。

---

## 10 预注册总表（正文 Table ICLR-1）

跑前把“预测”列填死。实测与状态跑完再填。

| 行 | 预言 | 推翻条件 |
|---|---|---|
| E1-syn-ind | 同 λ：axis ind ≤0.10，dense ≥0.80 | 两者 CI 重叠或符号反 |
| E1-rot | 功效 vs \(1-\lVert B^\top w\rVert_4^4\) 单调，r≥0.9 | 无单调 / 旋转连 label 一起垮 |
| E1-gpt2 | 复现轴 1.00 / ~0.45 / ≤0.10 | 层级消失 |
| E2-align | \(B_{\mathrm{align}}\) 上 ind→0，同一 \(w\) 在 \(B_{\mathrm{rand}}\) 上 ind 高 | 盲区不跟基走 |
| E2-coef | 4 范数系数赢过 ∞ / PR / \(\ell_1\)，Δr≥0.10 | 别的几何分数赢 |
| E3 | \(C_{\mathrm{PCA}}\) 比 \(C_{\mathrm{raw}}\) 高至少 0.05 | raw 不差 |
| E4-diag | axis→`cov_not_ind`，dense→`visible` | 类与几何无关 |
| E4-SAE-A/B | A：同 λ dense 恢复更高且旋转可救；B：无差则改口代理松 | 结果与 A、B 都不符还硬写 dark matter |
| E5 | 自然银行几何 ΔR²<0.05，强度/频率主导 | 几何仍是主效应 |
| E0 | 全部主相关带 95% CI | CI 跨 0 的相关不得当主验证 |

---

## 11 主张–实验–审稿人 映射

| 审稿人问题 | 用哪组答 | 正文怎么写 |
|---|---|---|
| 同强度为何分居边界两侧 | E1 | 主图左：一对柱。不是 λ，是 \(C_i\) |
| feature recovery 有何用 | E4 诊断类 | 三级诊断；miss 先看证据通道 |
| 怎么改善 SAE | E4 旋转干预 | 不改损失，改几何/加标签/换 nuisance；代理紧才预测恢复 |
| uniqueness / SOTA | E2 竞争预测子 + 拒绝 bake-off | 独特预言是基相对吸收，不是新 SAE |
| 对角模型在哪个坐标 | E2 + §1.3 | 默认 PCA 白化坐标；盲区跟 \(B\) 走 |
| 为何坐标方差 refit 是代理 | E2 家族陈述 + E4 A/B | 代理 = 固定基上不能造相关；E4 测紧不紧 |
| 各向异性？ | E3 | 白化后同一 \(C\)；PCA/ZCA 对照 |
| 已知方向 vs 无监督发现 | E1–E3 定理侧；E4 标成后果 | Table 4 上正文 |
| 自然事件远离轴盲区 | E5 | 做成结果：自然可见性由 λ/p/σ 驱动 |
| CI / 敏感性 | E0 / E6 | 表里加区间；附录网格 |
| 收窄 dark matter | E4-B 预注册 | Prop 6 保持理想化；SAE 只在 A 成立时写“操作后果” |

---

## 12 算力与日程

| 实验 | 估计 | 依赖 |
|---|---|---|
| E0 | CPU 0.1–1 h（有旧 CSV 则分钟） | 旧 artifacts；没有就重跑合成 545s |
| E1 合成+旋转 | CPU 1–3 h | 无 |
| E1/E2 GPT-2 | CPU 数小时（检测器，不训模型） | 旧 30k 白化 cache |
| E3/E5 | CPU <30 min | 旧 735 配置 + 47 events 表 |
| E4 TopK | 1 GPU × 2–6 h（24 配置级） | C16 脚本；无脚本则合成 TopK 也可先出正文图 |
| E6 | CPU 1–2 h | 同一 cache |
| **合计** | **约 1 个 GPU 下午 + 1 天 CPU** | 不新训 GPT-2，不训大 SAE |

建议顺序：E0 → E1 合成（当天出主图原型）→ E2 合成旋转 → E3/E5 复分析 → E1/E2 GPT-2 → E4 → E6。  
E1 合成失败则停，先修统计量，不要靠 GPT-2 救命。E4 若走预注册 B，仍然投，不改 E1–E3。

---

## 13 实现时的协议陷阱（写给动手的人）

1. **旋转必须只动 ind 的坐标，不动注入强度。** 检查：label 功效在 \(\theta\) 上平坦。否则图是假的。
2. **cov 的背景协方差必须来自无注入 cache**，不能在 \(H_1\) 上估。否则 cov 与 ind 会塌成同一个统计量。
3. **不要在 ind 里沿 \(w\) 做已知方向的二次型再减对角** 和“全矩阵 offdiag”混用而不声明。正文用 offdiag Frobenius，与旧稿 6.1 一致；已知方向的 \(w^\top\mathrm{offdiag}(\Sigma)w\) 可作附录稳健性，不得替换主统计量。
4. **GPT-2 的 axis 是白化后的 \(e_1\)**，不是 embedding 第 1 维。图注必须写。
5. **E4 的 SAE 恢复阈值抄 C16，不许对着新结果调。** 阈值进预注册表。
6. **自然事件方向仍用 train split 的 mean-difference，test 上算 z。** 不许用全量方向再报功效。
7. 若没有旧 cache：先把 E1/E2 合成 + E0 合成 CI 做成可投稿主图；GPT-2 桥可以晚 48h，但不要用合成冒充真实激活。

---

## 14 一句话收束

旧实验已经证明定律在匹配模型和 GPT-2 注入上出现。ICLR 要补的不是更多相关，而是：

**同一条弱特征，λ 钉死，只改检测器的基，可见性跟着 \(1-\lVert B^\top w\rVert_4^4\) 翻转；这个诊断类再去预测（或预测失败并照登）TopK 是否恢复。**

这同时回答 R1 的用途/同强度问题、szZ2 的坐标系/各向异性/过声称、以及 N2Qr 的“你到底在比什么”。
