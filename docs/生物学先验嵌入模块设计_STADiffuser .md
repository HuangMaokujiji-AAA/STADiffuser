# 空间转录组学 3D 重建中的生物学先验嵌入模块设计

> 面向 STADiffuser Graph Attention 改造  
> 目标：将转录调控（GRN）、信号通路、细胞通讯（CCI）先验直接嵌入注意力/图模型

---

## 一、生物学先验网络来源

在注入模型前，需明确三类先验的数据来源、构建方式与推荐使用策略。

### 1. 转录调控网络（Gene Regulatory Network, GRN）

| 来源 | 类型 | 推荐用途 | 备注 |
|------|------|----------|------|
| **ARACNe / ARACNe-AP** | 数据驱动推断 | 从当前或同源 scRNA/ST 数据推断细胞类型特异 GRN | 因果性较强，适合作为核心先验 |
| **SCENIC / SCENIC+** | 数据驱动 + motif | 转录因子–靶基因调控 | 可获得 TF 活性与 regulon |
| **ENCODE** | 实验验证（ChIP-seq 等） | 高质量 TF–靶基因边 | 保守、可信度高 |
| **NicheNet** | 整合先验 | 配体→靶基因调控效应 | 同时覆盖信号与调控 |
| **DoRothEA / RegNetwork** | 知识库 | 快速构建初始 GRN | 需过滤低置信度边 |

**建议处理流程**：
1. 优先使用细胞类型/状态特异的 ARACNe 或 SCENIC 结果。
2. 与 ENCODE 高置信度边取交集或加权融合。
3. 最终输出稀疏邻接矩阵 \(A_{\text{GRN}} \in \{0,1\}^{G \times G}\) 或带权重的置信度矩阵。

### 2. 信号通路 / 蛋白互作网络（Signaling / PPI）

| 来源 | 类型 | 推荐用途 | 备注 |
|------|------|----------|------|
| **STRING** | PPI + 功能关联 | 高置信度蛋白互作（score > 0.7） | 最常用，覆盖广 |
| **PrePPI** | 结构预测 PPI | 补充实验未覆盖的互作 | 适合结构相关信号 |
| **KEGG / Reactome** | 通路知识库 | 通路内部基因集合与有向边 | 用于通路活性计算 |
| **OmniPath** | 整合信号资源 | 激酶–底物、信号转导边 | 质量较高 |

**建议处理流程**：
- 以 STRING high-confidence 边为主干。
- 用 KEGG/Reactome 标注通路归属，便于后续通路水平损失或条件注入。
- 可构建 gene–gene 或 pathway–pathway 两级图。

### 3. 细胞–细胞通讯网络（Cell–Cell Communication, CCI）

| 来源 | 类型 | 推荐用途 | 备注 |
|------|------|----------|------|
| **CellChatDB** | 配体–受体对 | 经典 CCI 分析 | 覆盖较全 |
| **NicheNet** | 配体–靶基因 | 连接细胞间信号与下游调控 | 与 GRN 天然衔接 |
| **CellPhoneDB** | 配体–受体 | 单细胞/空间常用 | 含复合物信息 |
| **CellTalkDB / ICELLNET** | 补充资源 | 扩展配体–受体对 | 可选 |

**空间加权方式**（关键）：
\[
w_{ij}^{\text{CCI}} = \text{LR_score}(i,j) \times \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
\]
其中 \(d_{ij}\) 为 spot/cell 间的空间（或 3D）距离，\(\sigma\) 控制通讯有效半径。

### 4. 先验使用总原则

- **稀疏化**：只保留高置信度边，避免计算爆炸。
- **层级化**：Gene 级 → Module/Pathway 级 → Spot 级（通过表达加权聚合）。
- **动态可调**：预训练阶段强约束先验，微调阶段允许数据驱动修正（见模块 D）。
- **多视图共存**：空间邻近图 + GRN 图 + CCI 图同时存在，由注意力或门控决定权重。

---

## 二、四个核心模块设计

### 模块 A：Graph Diffusion Kernel Attention（GDKA）

**设计目标**  
将分子互作图的结构信息通过扩散核直接嵌入 Transformer / GAT 的 Query，实现“生物学路径约束的注意力”。

**核心思想**（源自 GREmLN / Califano 组）  
1. 从先验邻接矩阵 \(A\) 构建图拉普拉斯 \(L = D - A\)。  
2. 计算扩散核 Gram 矩阵：
   \[
   K = \exp(-\beta L)
   \]
   （实际可用 Chebyshev 多项式近似加速）。  
3. 用 \(K\) 变换 Query：
   \[
   Q' = K Q
   \]
4. 执行标准注意力：
   \[
   \text{Attention}(Q', K, V)
   \]

**在 STADiffuser 中的插入位置**  
- Stage I Graph Attention 层：对 Gene 维度或 Spot 维度施加 GDKA。  
- 可选：在 Self-Attention Block 中替换标准自注意力。

**优点**  
- 长程依赖被生物学网络结构软约束。  
- 参数效率高，符合“pre-wire biological knowledge”理念。  
- 可解释性强（注意力流沿着先验路径）。

**实现要点**  
- \(\beta\)（扩散速率）和 Chebyshev 阶数需调参。  
- 先验图需对称化或处理为无向图的随机游走形式。

---

### 模块 B：Prior-Biased Multi-Head Attention

**设计目标**  
在标准注意力分数上引入生物学先验作为乘法或加法偏置，实现轻量、易插拔的改造。

**核心公式**
\[
\alpha_{ij} = \text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d}} \times P_{ij}^{\text{prior}} + b_{ij}^{\text{spatial}} \right)
\]
或对数形式：
\[
e_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + \lambda \log(P_{ij}^{\text{prior}} + \epsilon)
\]

其中 \(P_{ij}^{\text{prior}}\) 来自：
- GRN 边权重
- 配体–受体表达乘积 × 空间距离衰减
- 通路共归属分数

**在 STADiffuser 中的插入位置**  
直接替换原有 Graph Attention Layer 中的 \(\alpha_{ij}\) 计算。

**优点**  
- 实现简单，几乎不改变原有代码骨架。  
- 可学习缩放系数 \(\lambda\) 控制先验强度。  
- 与多头机制天然兼容。

**推荐作为第一阶段快速验证模块。**

---

### 模块 C：Hierarchical Message Passing（空间层 + 生物学层）

**设计目标**  
显式分离不同生物学尺度的信息流，实现跨层级交互。

**结构**

1. **Intra-spot / Gene 层**  
   - 在每个 spot 内部，用 GRN / PPI 图进行消息传递。  
   - 更新基因表征，捕捉细胞内调控逻辑。

2. **Inter-spot / Spatial + CCI 层**  
   - 用空间邻近图 + 配体–受体图进行消息传递。  
   - 更新 spot / cell 表征，捕捉微环境通讯。

3. **Cross-level Attention**  
   - Gene 表征与 Spot 表征通过 Cell–Gene Attention 交互（可参考 MAGNET 设计）。  
   - 实现“细胞内状态 ↔ 细胞间通讯”的闭环。

**在 3D 重建中的价值**  
- 虚拟切片生成时，基因表达插值同时受空间平滑与调控网络约束。  
- 天然支持 3D 细胞通讯景观的推断。

**实现建议**  
- 使用异质图神经网络（Heterogeneous GNN）框架。  
- 不同边类型使用独立的消息函数与注意力头。

---

### 模块 D：动态先验精炼（Dynamic Prior Refinement）

**设计目标**  
解决先验噪声与不完整性问题，允许模型在数据驱动下修正先验。

**核心机制**
\[
A_{\text{final}} = \sigma(g) \odot A_{\text{prior}} + \bigl(1 - \sigma(g)\bigr) \odot A_{\text{learned}}
\]
其中 \(g\) 为可学习门控参数（可为边级或全局）。

**训练策略**（呼应 SpaGTL 思想）  
- **预训练阶段**：强约束 \(\mu > 0\)，迫使模型贴近先验。  
- **微调阶段**：降低或关闭强约束（\(\mu \to 0\)），允许上下文特异修正。

**附加技巧**  
- 用注意力权重反向更新先验边的置信度。  
- 对高频使用且与数据一致的边提高权重，对冲突边降权。

**优点**  
- 兼顾知识引导与数据适应性。  
- 降低对先验质量的绝对依赖。

---

## 三、模块组合推荐策略

| 阶段 | 推荐模块组合 | 目标 |
|------|--------------|------|
| 快速验证 | 模块 B | 最小改动验证先验有效性 |
| 核心创新 | 模块 A + 模块 B | 深度嵌入 + 灵活偏置 |
| 完整系统 | 模块 A + C + D | 层级建模 + 动态精炼 |
| 3D 重建强化 | 模块 C 为主，辅以 A/B | 突出空间与通讯的跨层交互 |

---

## 四、与 STADiffuser 的对接要点

1. **Stage I（Graph Attention Autoencoder）**  
   - 优先改造 Graph Attention Layer（模块 A 或 B）。  
   - 图构建从单一 SNN 扩展为多视图（空间 + GRN + CCI）。

2. **Stage II（Latent Diffusion）**  
   - 可将先验网络的 embedding 或扩散核特征作为额外 condition 输入空间去噪网络。

3. **损失函数补充**  
   - 网络一致性损失（Network Consistency Loss）  
   - 配体–受体一致性损失（CCI Loss）  
   - 与原有重建损失、Triplet Loss 加权组合。

---

## 五、参考文献与资源（核心）

- GREmLN（Graph Diffusion Kernel Attention 思想）  
- STADiffuser 原论文（Graph Attention Autoencoder 结构）  
- SpaGTL、MAGNET、SMOReg（多视图图与先验约束实践）  
- NicheNet、CellChat、STRING、ARACNe、SCENIC（先验数据来源）

---

**文档版本**：v1.0  
**适用课题**：空间转录组学 3D 重建中的生物学先验网络嵌入  
**维护建议**：根据实际使用的先验数据库版本与 STADiffuser 代码更新同步修订
