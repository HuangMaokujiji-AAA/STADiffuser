# STADiffuser OpenST-3D 联合切片留出

在项目根目录的 STADiffuser Python 环境运行。正式训练需要 CUDA GPU，以及与 PyTorch 匹配的 PyG 邻居采样后端（`pyg-lib` 或 `torch-sparse`）。若使用 Diffusers 0.17.1，可搭配 `huggingface-hub==0.25.2`。

```bash
python -u -m experiments.openst_holdout \
  --input /data/Open-ST/GSE251926_metastatic_lymph_node_3d.h5ad \
  --device cuda:0 --output-dir output/openst_3d
```

按 `obs['n_section']` 将 **#4、#9、#18、#24** 一起留出。其余 15 张用于同一个模型的训练和 Seurat HVG 拟合，按合并后的训练细胞筛出 2,000 个基因。模型直接使用文件中已归一化的 log1p `X`。预测以目标的已配准 XYZ 为条件，使用 DLPFC 教程的无类别标签分支。

沿用 DLPFC 教程的 SpaAE（`gat_dim=[512, 32]`）、500 轮和批量 256，扩散训练 1,000 轮、批量 512、学习率 `1e-4`、权重衰减 `1e-6`、1,000 个 DDPM 时刻，邻域采样 `[5, 3]`，图半径 150 µm，XY 按 125 µm 量化。三维 denoiser 使用作者 T4 教程的独立 Z 编码，并按真实物理切片间距量化 Z。训练潜变量用图邻域小批量编码，预测时按目标整片的图解码。

输出目录包含 `hvg.csv`、`split.csv`、`run.json` 以及四个预测文件。文件的 `X` 是 2,000 个基因的模型表达预测，`obs_names` 保留原细胞 ID，`obsm['spatial']` 和 `obsm['spatial_3d_aligned']` 分别存配准 XY 和物理 XYZ。

| 文件 | 全量细胞数 |
| --- | ---: |
| `predicted_section_04.h5ad` | 77,868 |
| `predicted_section_09.h5ad` | 54,899 |
| `predicted_section_18.h5ad` | 80,288 |
| `predicted_section_24.h5ad` | 68,177 |

可以追加 `--prepare-only` 先核对完整数据划分并导出 HVG。使用 `--device cpu --smoke` 可抽每张 48 个细胞做一轮短跑。每次运行指定一个新的输出目录。
