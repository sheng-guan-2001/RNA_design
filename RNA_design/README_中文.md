# RNA_design

区域感知mRNA多目标同义序列设计：局部结构搜索与双条件CodonTransformer生成。

## 1. 安装

Python 3.10，PyTorch依赖对应CUDA 12.1。

```bash
pip install -r requirements.txt
```

另需 [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) 的 RNAfold、RNAplfold 命令行工具（若未安装，可用 `conda install -c conda-forge -c bioconda viennarna`）。

## 2. 数据下载

- [CodonTransformer 数据集](https://huggingface.co/datasets/adibvafa/CodonTransformer)：dataset.csv，也可从 [Zenodo](https://zenodo.org/records/12509224) 下载。
- [ColiFormer 数据集](https://huggingface.co/datasets/saketh11/ColiFormer-Data)：CAI.csv、finetune_set.json 及测试数据。
- [CodonTransformer 预训练权重及分词器](https://huggingface.co/adibvafa/CodonTransformer)。
- [大肠杆菌 CAI 参考序列来源](https://github.com/Benjamin-Lee/CodonAdaptationIndex)：示例 ecol.heg.fasta，需与原实验文件核对。

复制 `local.env.example` 为 `local.env`，修改数据和模型路径。ColiFormer 审核后的训练候选、物种审核及保留测试 CSV 不是直接下载的上游文件，可使用原实验文件，或通过 `create_train_dataset/audit_coliformer_training.py --help` 查看重建参数。严格复现须保留原测试集划分。

## 3. 运行顺序

```bash
bash scripts/01_prepare_data.sh
bash scripts/02_train.sh
bash scripts/03_optimize_prefix.sh
bash scripts/04_generate.sh
bash scripts/04_generate.sh --execute
bash scripts/05_score.sh
```

第一步处理数据；第二步训练 M4；第三步优化局部前缀；第四步生成同义后缀；第五步评分。已有处理好的数据可设置 PREP 跳过第一步。训练后将 CHECKPOINT 指向选定的模型。第四步先运行 `bash scripts/inspect_v3.sh` 核实原始 V3 生成入口并设置 V3_MODULE；首次运行只预览命令，确认后加 --execute。若生成文件名不同，设置 GENERATED_CSV。原始目标配置可用时保存在 configs 中，LOCAL_TARGETS 与 FINAL_TARGETS 指向相应文件。严格复现仍需核实原始 V3 搜索参数。

输出默认保存在 runs/reproduction_01/，不包含原始数据或权重。保留 CodonTransformer 原始许可及署名。
