# 大学生自杀意图风险预测模型 | Suicidal Intent Prediction Model

## 项目简介 | Introduction

本研究基于中国大学生心理健康筛查量表（CCSMHSS）普查数据，构建并验证了一套可解释机器学习框架，用于识别大学生自杀意图，以弥补自陈量表直接测查的社会期许偏差，并处理混合特征数据的类别不平衡问题。研究纳入 3068 份有效样本，系统考察了 21 项心理健康维度与自杀意图的关联及共线性，采用 SMOTE-ENC 算法处理混合数据的类别不平衡，对比 6 种机器学习模型的预测效果，并通过 SHAP 框架从全局与个体层面解释模型决策。结果显示，逻辑回归为最优模型：经数据增强后召回率提升至 0.981，漏检率降至 1.9%，AUC-ROC 达 0.846，与集成模型区分度无统计学差异；幻觉妄想、抑郁、躯体化、自伤行为与敌对攻击为核心预测因子。该框架可作为高校心理普查的辅助筛查工具，为精准干预提供靶点依据。

This study constructs and validates an interpretable machine learning framework for identifying suicidal intent among college students based on the Chinese College Student Mental Health Screening Scale (CCSMHSS) census data, to compensate for social desirability bias in direct self-report assessment and address class imbalance in mixed-type feature data. Based on 3,068 valid samples, the study systematically examines the associations and multicollinearity between 21 mental health dimensions and suicidal intent, applies the SMOTE-ENC algorithm to handle class imbalance in mixed data, compares the predictive performance of 6 machine learning models, and interprets model decisions at both global and individual levels via the SHAP framework. Results show that logistic regression is the optimal model: after data augmentation, recall increases to 0.981 with a missed-detection rate of 1.9% and an AUC-ROC of 0.846, with no statistically significant difference in discriminative ability from ensemble models. Hallucination-delusion, depression, somatization, self-harm behavior and hostility are core predictors. This framework can serve as an auxiliary screening tool for college mental health census and provide target evidence for precise intervention.

## 数据说明 | Data Description

- 样本为高校新生心理健康普查数据，包含人口学特征与 21 项心理健康维度指标。
- 经一致性过滤后纳入有效样本 3068 例，自杀意图阳性占比 17.18%。
- 原始数据因隐私保护不存储于本仓库，仅保留分析代码与聚合结果。
- Samples are from college freshman mental health census, including demographic features and 21 mental health dimension indicators.
- 3068 valid samples are retained after consistency filtering, with 17.18% positive suicidal intent.
- Raw data are not stored in this repository for privacy protection; only analysis code and aggregated results are provided.

## 仓库结构 | Repository Structure

```
suicidal-intent-prediction-smote-enc-shap/
├── code/
│   ├── main_pipeline.ipynb      # 完整分析流水线：数据清洗→建模→评估→可解释性分析（最优参数，一键复现）
│   ├── model_tuning.ipynb       # 网格搜索调参：六模型超参数寻优（5折交叉验证，召回率优先）
│   └── check_versions.py        # 依赖库版本检测脚本
├── results/
│   ├── tables/                # 结果表格（CSV）
│   │   ├── table1_population.csv          # 人口学分布
│   │   ├── table2_point_biserial.csv      # 点二列相关
│   │   ├── table3_smote_comparison.csv    # 增强前后对比
│   │   ├── table4_delong.csv              # DeLong 检验
│   │   └── table5_vif.csv                 # VIF诊断
│   └── figures/               # 结果图（PNG）
│       ├── fig1_spearman_heatmap.png
│       ├── fig2_recall_comparison.png
│       ├── fig3_roc_curves.png
│       ├── fig4_shap_beeswarm.png
│       ├── fig5_shap_waterfall.png
├── data/                        # 原始数据表头
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## 环境配置 | Environment Setup

1. Python 版本要求：3.13+
2. 安装依赖：
```
pip install -r requirements.txt
```
3. 环境校验：运行版本检测脚本，确认所有依赖安装正确
```
python code/check_versions.py
```

## 使用说明 | Usage

### 1. 复现

 `code/main_pipeline.ipynb`，为论文代码，由于原始数据无法公开可以将代码中的数据集替换：

- 数据读取、清洗、特征工程与分层划分
- SMOTE-ENC 不平衡数据增强
- 六模型训练与性能指标计算
- ROC 曲线、召回率柱状图绘制
- DeLong 检验、校准分析
- SHAP 全局与个体可解释性分析

### 2. 自定义调参
`code/model_tuning.ipynb`，可修改参数网格对 6 种模型重新进行网格搜索调优；调优目标为 5 折交叉验证召回率最大化，支持自定义参数范围。

### 3. 替换自有数据

1. 将你的 Excel 数据文件放入 `data/` 文件夹
2. 修改 `main_pipeline.ipynb` 开头的文件路径列表与 sheet 名配置
3. 按需调整一致性过滤阈值、目标变量定义、特征列规则即可适配其他样本

## 主要结果 | Key Results

- 经 SMOTE-ENC 增强后，最优逻辑回归模型召回率提升至 0.981，漏检率降至 1.9%，AUC-ROC 为 0.846
- 模型区分度与 XGBoost、LightGBM 等集成模型无统计学差异
- 核心预测因子：幻觉妄想、抑郁、躯体化、自伤行为、敌对攻击
- 人口学特征对自杀风险预测的边际贡献极低
- After SMOTE-ENC augmentation, the optimal logistic regression model achieves recall of 0.981, missed-detection rate of 1.9%, and AUC-ROC of 0.846.
- No statistically significant difference in discriminative ability between logistic regression and ensemble models (XGBoost, LightGBM).
- Core predictors: hallucination-delusion, depression, somatization, self-harm behavior, hostility.
- Demographic features have negligible marginal contribution to suicide risk prediction.

## 免责声明 | Disclaimer

本项目仅用于学术研究与方法学探索，不构成临床诊断建议。模型输出为风险排序参考，不能替代专业心理评估与临床诊断。

This project is for academic research and methodological exploration only, and does not constitute clinical diagnosis advice. Model outputs are for risk ranking reference only and cannot replace professional psychological assessment and clinical diagnosis.
