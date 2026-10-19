# MERFISH mouse hypothalamus: leave-one-slice-out

入口是 `docs/source/tutorials/T5-MERFISH-leave-one-slice-out.ipynb`。Notebook 可直接读取原始五个 h5ad，参数集中在开头；数据处理和训练实现位于 `experiments/merfish_holdout.py`。不修改 T4 或 STADiffuser 核心模型。

## 输入与三次实验

| 文件 | 原始 slice_id / z | 细胞数 | 用法 |
| --- | ---: | ---: | --- |
| merfish_0.h5ad | 0.04 | 5488 | 每次均用于训练 |
| merfish_1.h5ad | 0.09 | 5557 | 第一次留出 |
| merfish_2.h5ad | 0.14 | 5926 | 第二次留出 |
| merfish_3.h5ad | 0.19 | 5803 | 第三次留出 |
| merfish_4.h5ad | 0.24 | 5543 | 每次均用于训练 |

每折从头初始化自编码器和扩散模型，使用另外四张切片训练。共同基因面板为 155 个基因。这里采用**已知留出细胞坐标上的表达重建**：目标细胞位置是输入，目标表达与真实注释均不是预测输入；这不是预测未知细胞数量/位置的任务。

## 转换与预处理

- 转换保留 `X`，将原始二维 `obsm['spatial']` 存入 `spatial_xy`，追加数值 `obs['slice_id']` 作为第三维。统一字段为 `obs['slice_ID']`，细胞名带源切片前缀。
- 保留文件中提供的 XY，不进行配准；输入需处于可比较的共同 XY 坐标系。Notebook 的叠图供检查，居中范围相近不能证明已经配准。
- 源数据 X 非整数、非负，没有 counts layer 或明确变换说明。默认在**分折后**做 `log1p(X)`，不再进行 library-size normalization。`expression_transform` 指训练变换，输出 X 保持模型训练尺度；主指标统一使用下述周报 log 表达尺度，不称作原始 UMI 计数。若另有证据表明源 X 已是 log 表达，需要同时明确跨方法评价尺度，不能仅凭取值较小就判断。
- 默认保留全部共同基因，不在全数据上选 HVG、PCA 或训练标准化器。潜空间 MinMax 参数仅用四张训练切片拟合。
- 原始 Z 的物理单位未提供。配置 `xy_division=20`、`z_division=0.01` 是模型坐标量化尺度，不代表确认了物理单位。每折只用训练坐标拟合一个共同原点，并用同一原点转换留出坐标。坐标限制在模型位置编码的 0..999 范围。
- 每张切片独立构造 8 邻居图，不连接训练和留出细胞。脚本直接用 SciPy 构图，避免核心 `utils._cal_spatial_net` 的 kNN 距离索引问题；核心文件未改动。

## 条件标签

默认 `condition='spatial_knn'`：训练使用源数据的 `annotation`（本数据中是解剖区域，**不是 cell_class**），预测阶段用训练坐标及其 annotation 的距离加权 15-NN 推断条件。空间距离使用配置中的三轴缩放；不调用留出真实标签、表达或表达派生嵌入。扩散模型显式设置 `num_class_embeds`。

因此，默认结果名为 `STADiffuser_predicted_condition`，对应“STADiffuser + 训练坐标邻居推断条件”的适配协议，不能当作作者已经报告的 MERFISH 数值复现。`label_key='cell_class'` 可改变条件语义，需要新 RUN_NAME。`condition='none'` 是显式命名的 `STADiffuser_spatial_only` 变体；不自动把真实留出注释作为条件。

稀有细胞评价单独使用 `cell_type_key='cell_class'`，不把解剖区域 annotation 混作细胞类型。STADiffuser 当前没有直接生成 cell_class，因此每折用**训练切片的 log 表达与 cell_class 建立 1-NN 标注器**，再标注生成表达，保存在 prediction.obs['cell_class']，来源为 `cell_type_source='training_expression_1nn'`。不复制留出真实标签，也不把空间邻居推断的扩散条件当作细胞类型预测。稀有类指标评估的是生成表达加该标注步骤，受标注器影响。

## 训练、预测与评估

沿用 T4 的 `SpaAE(input_dim=155, gat_dim=[512,32], block_out_dims=[32,32], block_list=['AttnBlock'])` 和三维 `SpaUNet1DModel(in_channels=18, spatial_encoding='sinusoidal3d', spatial3d_concat=True)`。默认自编码器预训练 200 轮、triplet 对齐 300 轮、扩散训练 500 轮；triplet margin=1，学习率 1e-4，MNN 邻居 15，每 50 轮更新。训练 DDPM 使用 1000 个时刻。批量为 128；对齐阶段核心实现仍使用整张训练切片。

每折训练种子为 2024；默认生成种子为 2024、2025、2026，每次 1000 步。只将去噪过程分成 512 个细胞的小批量；图解码对整张目标切片执行，保留解码器的整图行为。三次生成的均值用于主结果，同时保存每个生成种子的指标；生成种子重复不等同于训练重复。

## 按 2026.10.18 学习周报评价

只评价 STADiffuser，不生成或比较简单插值结果。指标口径依据用户提供的学习周报，并核对对应 DeepSpatial 留出 Notebook 的评估实现。

**主表达指标**先对每个细胞得到 `log1p(max(源 X,0))`，再沿细胞维度求每个基因的均值，得到 155 维向量。`mse`、`mae` 是真值/预测均值向量的均方误差、平均绝对误差；`pcc` 是两个均值向量的 Pearson 相关系数。不是逐元素表达 MSE。若模型训练/输出已在 log1p 空间，评价只把负值裁为 0，不重复 log；若配置为 none，则评价时执行 log1p。原预测文件保留未经裁剪的模型输出，另报告负值比例。

`pointwise_mse` / `pointwise_mae`、按基因跨细胞相关和按细胞跨基因相关保留为辅助诊断，不混入周报的主指标。`gene_metrics.csv` 保存每基因的真值/预测平均表达与均值误差，便于追溯 MSE。常数向量的相关性记 NaN。

**稀有类**由每折训练四片中 cell_class 的占比严格小于 1% 定义，训练计数为 0 的类别单列 unseen_in_training。对真值中存在的每个训练稀有类，计算 XY 平面 50/100 µm 内存在同类预测点的比例，然后对类别等权平均，分别输出 `rare_recall_50um`、`rare_recall_100um`；同时保留 `rare_macro_coverage_recall_*um` 同义字段。整类没有生成时召回率为 0，没有预测点时精确率为 NaN。它是点集覆盖，允许多个真值点共用邻居，不是唯一匹配召回。

逐类保存训练/真值/预测数量、比例差、同类最近邻距离、类内平均 log 表达 PCC、覆盖召回和精确率。默认额外提供 25 µm 半径。`xy_unit_um=1` 按周报将本数据源 XY 解释为 µm；可显式修改换算系数，50/100 µm 必须保留。

**已知坐标任务边界**：STADiffuser 此实验输入了真实目标细胞的数量和位置，所以整体数量误差与不区分类别的双向最近邻距离固定为 0，仅记录为协议检查，不能据此宣称模型恢复了细胞数量/位置。稀有类同类覆盖仍取决于生成表达推断的类型。周报中的 DeepSpatial 自由生成目标点集，输入条件不同，这些固定值不能作公平排名。

三折汇总采用切片等权平均及样本标准差；训练集重叠，标准差不是独立重复的置信区间。逐类汇总另给三折真值/预测数量合计与每折指标均值。真实注释只在生成完成后用于评价。

`SMOKE_TEST=True` 使用每片 48 个细胞、每阶段 1 轮、10 步采样，输出到独立的 smoke 目录；这是工程检查，不能用于重建性能结论。减少正式预算或细胞数的运行在指标中记为 `development`。

## 本地转换（可选）

Notebook 不要求预先转换。若只想传一个文件到服务器，在项目根目录执行：

```bash
python experiments/merfish_holdout.py --data-dir /path/to/merfish_mouse_hypothalamus --output data/merfish_mouse_hypothalamus_3d.h5ad
```

该步骤只需要 anndata、numpy、pandas、scipy，不依赖训练模型或 GPU。合并文件约含 28,317 x 155 的表达矩阵，保持源 X 未变换。Notebook 可读取原始文件目录，也可设置 `SOURCE_H5AD` 使用该合并文件。

## 服务器运行

上传整个项目的新增文件以及原始五个 h5ad（或合并 h5ad）。Notebook 的 `DATA_DIR` / `SOURCE_H5AD` 指向服务器位置，使用已跑通 T4 的环境。先运行数据加载与分折检查，再正式训练。

如需后台运行，从项目根目录在 tmux 会话中导出并执行：

```bash
python -m jupyter nbconvert --to script docs/source/tutorials/T5-MERFISH-leave-one-slice-out.ipynb --output merfish_loso --output-dir .
python -u merfish_loso.py > merfish_loso.log 2>&1
```

每折完成后写出 `metrics.csv`。相同数据和训练配置下会复用已完成折；旧预测缺少新版周报指标时自动重算，不重复训练。设置 `EVALUATE_ONLY=True` 可只读取已有 prediction.h5ad 重算，不需要模型/GPU。旧版仅保存平均预测，所以重算只产生均值预测的指标；单次生成矩阵未保存，不伪造其新版指标。修改评估半径/稀有阈值可重算，修改模型/训练配置或数据必须换 RUN_NAME。未完成训练保留已有权重并要求换 RUN_NAME，当前没有优化器/阶段中断续训。

## 输出

每个 `output/<RUN_NAME>/merfish_i/` 包含：`run_config.json`、`evaluation_config.json`、`split.json`、训练/查询细胞名单、`train_cell_counts.csv`、`environment.json`、`autoencoder.pt`、`denoiser.pt`、`latent_normalizer.npz`、`losses.npz`、`prediction.h5ad`、`truth.h5ad`、`gene_metrics.csv`、`cell_metrics.csv`、`cell_type_metrics.csv`、`rare_cell_metrics.csv` 和 `metrics.csv`。PT 文件保存 state_dict。模型结构、基因顺序和坐标/类别映射可从 Notebook 配置、prediction.var_names 和 split.json 恢复；不与果蝇模型权重混用。

Notebook 另写三折总表 `all_metrics.csv`、`fold_mean_metrics.csv`、`summary.csv`、`all_cell_type_metrics.csv`、`all_rare_cell_metrics.csv`、`rare_class_summary.csv`，显示真实/预测两列的空间表达和稀有类明细。没有预填正式训练结果或准确率。

最小检查：`python -m unittest discover -s tests -p test_merfish_holdout.py`。覆盖留出表达/注释隔离、训练/测试图隔离、共同坐标、均值向量 MSE 与逐元素 MSE 的区别、避免二次 log、稀有类整类缺失召回记零、训练表达标注和常数相关系数处理。

参考：[作者 T4 教程](https://stadiffuser.readthedocs.io/en/latest/tutorials/T4-3D-slice-modeling-Drosophila-embryo.html)。
