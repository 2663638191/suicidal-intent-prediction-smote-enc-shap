# 大学生自杀意念筛查的可解释机器学习代码

[English](README.md)

本仓库提供大学生心理筛查研究的完整中文分析代码，以及英文运行入口、标签和结果导出工具。公开内容包括源码、空白表头、字段说明和必要运行文档。

## 目录

| 位置 | 内容 |
|---|---|
| [code/en](code/en/) | 英文运行入口、导出工具及计算实现 |
| [code/zh](code/zh/) | 中文完整源码 |
| [data/schema](data/schema/) | 空白表头与字段字典 |
| [docs/data.md](docs/data.md) | 输入定义与数据可用性说明 |
| [docs/reproducibility.md](docs/reproducibility.md) | 环境、命令与检查 |
| [tests](tests/) | 基于公开元数据及合成输入的检查 |

## 运行

使用 Python 3.13，创建并激活虚拟环境后安装依赖：

```bash
git clone https://github.com/2663638191/suicidal-intent-prediction-smote-enc-shap.git
cd suicidal-intent-prediction-smote-enc-shap
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`，macOS/Linux 使用 `source .venv/bin/activate`。然后安装依赖：

```bash
python -m pip install -r requirements.txt
python code/en/run_analysis.py --help
```

准备与[输入结构](data/schema/input_schema.json)一致且已获授权的数据后运行：

```bash
python code/en/run_analysis.py --fit --data-dir data/private/original --output results/generated --workers 4
```

中文运行命令及其他输出选项见[中文代码说明](code/zh/运行说明.md)和[运行指南](docs/reproducibility.md)。

## 数据可用性

原始个体筛查数据受数据提供方授权和保密要求限制，不公开提供。表头和字段字典用于说明输入结构，不等于公开数据集；复算论文中的实测数值需要获得原始数据的合法使用授权。

## 检查

```bash
python -m unittest discover -s tests -v
```

公开检查使用源码、元数据和合成输入。代码保留表格与图片生成功能，当前公开范围不包含本研究的实证输出文件和论文图件。

## 引用与许可

通过 [CITATION.cff](CITATION.cff) 或 GitHub 的 **Cite this repository** 引用软件，并记录所用提交版本。代码沿用 [MIT 许可证](LICENSE)。修改建议见 [CONTRIBUTING.md](CONTRIBUTING.md)，仓库更新见 [CHANGELOG.md](CHANGELOG.md)。
