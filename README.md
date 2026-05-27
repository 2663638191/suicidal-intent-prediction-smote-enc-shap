# 不平衡数据下自杀意图的可解释机器学习预测框架 | Suicidal Ideation Prediction with SMOTE-ENC and SHAP

## 论文信息 | Paper Information
**标题 | Title**: Construction of Interpretable Machine Learning Prediction Framework and Identification of Influence Factors for Suicidal Intent Under Imbalanced Data Conditions  
**作者 | Authors**: Liu He, Zhi Yin*  
**机构 | Affiliation**: School of Statistics and Data Science, Ningbo University of Technology  
**代码仓库 | Repository**: https://github.com/2663638191/suicidal-intent-prediction-smote-enc-shap
---

## 摘要 | Abstract

### 中文摘要
自陈量表中自杀意图条目的直接测查受社会期许偏差限制，且自杀相关数据存在天然类别不平衡、传统预测模型可解释性不足的问题。本研究基于6272份高校新生中国大学生心理健康筛查量表（CCSMHSS）实测数据，构建了一套可解释的机器学习预测模型，采用Synthetic Minority Oversampling Technique-Encoded Nominal and Continuous（SMOTE-ENC）算法处理不平衡数据，结合SHapley Additive exPlanations（SHAP）框架开展可解释性分析。结果显示，逻辑回归为最优模型，经数据增强后阳性样本召回率从0.772提升至0.962，AUC-ROC稳定为0.820，识别出幻觉妄想、抑郁等5项核心预测因子。该模型可有效弥补心理筛查中仅依赖被试对自杀意图相关条目直接作答的效度局限，为高校精准自杀风险筛查提供数据支撑。

### English Abstract
The direct assessment of suicidal intent items in self-report scales is constrained by social desirability bias, and suicide-related data suffer from inherent class imbalance and insufficient interpretability of conventional predictive models. Based on the actual survey data of 6,272 incoming college students from the Chinese College Student Mental Health Screening Scale (CCSMHSS), this study constructed an interpretable machine learning predictive model, employed the Synthetic Minority Oversampling Technique-Encoded Nominal and Continuous (SMOTE-ENC) algorithm to handle imbalanced data, and combined it with the SHapley Additive exPlanations (SHAP) framework to conduct interpretability analysis. The results show that the logistic regression model is the optimal one. After data augmentation, the recall rate of positive samples increased from 0.772 to 0.962, and the AUC-ROC remained stable at 0.820. Five key predictive factors such as hallucination-paranoia, depression were identified. This model can effectively remedy the validity limitation of psychological screening that relies solely on the direct responses of the participants to the suicidal intent items, providing data support for precise suicide risk screening in colleges.

---

## Overview
This study addresses three key challenges in psychological screening:
1. The SMOTE-ENC algorithm was applied to the mixed feature psychological measurement data, solving the inherent class imbalance problem of suicide data.
2. It was verified that logistic regression was the optimal model in the CCSMHSS data set of this study, achieving a high-risk sample recall rate of 96.2%.
3. Through the SHAP explainability framework, the core predictors of college students' suicidal intentions were clarified, providing a scientific basis for targeted intervention.

---

## 数据可用性 | Data Availability
重要声明 | Important Notice：本研究使用的 6272 份高校新生心理健康普查原始数据涉及个人隐私，受严格的伦理审查和法律保护，无法公开共享。
仓库中仅提供data/sample_data.csv作为数据格式示例，您需要使用自己的数据集按照相同格式整理后运行代码。
The original mental health census data of 6,272 college freshmen used in this study involves personal privacy, protected by strict ethical review and legal regulations, and cannot be publicly shared.
Only data/sample_data.csv is provided in the repository as a data format example. You need to use your own dataset organized in the same format to run the code.

---

## 结果 | Results
所有论文中的结果图已保存在figures/文件夹中，与论文发表版本完全一致：
ROC_Curves.png：6 种机器学习模型的 ROC 曲线对比
Recall_Comparison.png：SMOTE-ENC 数据增强前后各模型召回率对比
SHAP_Beeswarm.png：逻辑回归模型的全局特征重要性蜂群图
SHAP_Waterfall_179.png：典型高风险样本的个体水平解释瀑布图
All result figures in the paper are saved in the figures/ folder, completely consistent with the published version of the paper.

---

## 量表说明 | CCSMHSS Scale Description
量表介绍 | Introduction
本研究使用中国大学生心理健康筛查量表（Chinese College Student Mental Health Screening Scale, CCSMHSS），这是中国教育部统一推广使用的标准化高校新生心理健康普查工具。
该量表采用三级筛查体系设计，共包含 96 个题目，全面覆盖大学生心理健康的各个维度：
一级筛查：严重心理危机（自杀意图、幻觉妄想）
二级筛查：内化性心理问题 + 外化性心理问题（共 18 个维度）
三级筛查：一般压力与适应困扰（共 5 个维度）
完整的量表题目和维度对应关系请查看：data/CCSMHSS_Scale_Description.xlsx

This study adopts the Chinese College Student Mental Health Screening Scale (CCSMHSS), a standardized mental health census tool for Chinese college freshmen uniformly promoted by the Ministry of Education of the People's Republic of China.
This scale is designed with a three-level screening system and contains a total of 96 items, covering all dimensions of Chinese college students' mental health comprehensively:
Level 1 Screening: Severe psychological crisis (suicidal ideation, hallucination and delusion)
Level 2 Screening: Internalized psychological problems + Externalized psychological problems (18 dimensions in total)
Level 3 Screening: General stress and adjustment distress (5 dimensions in total)
For the complete correspondence between scale items and dimensions, please refer to: data/CCSMHSS_Scale_Description.xlsx

## 数据格式说明 | Data Format
本研究使用各维度的标准分作为模型输入特征：
连续特征：22 个维度的标准分（Z-score 标准化）
分类特征：性别、民族、生源地、是否独生子女
目标变量：自杀意图（Suicide_Intent），二分类标签（1 = 阳性，0 = 阴性）

This study uses the standard scores of each dimension as the input features for the model:
Continuous features: 22 standard scores of dimensions (Z-score standardized)
Categorical features: gender, ethnicity, place of origin, whether an only child
Target variable: Suicide_Intent, a binary label (1 = positive, 0 = negative)

## Requirements
- Python 3.13.5
- scikit-learn 1.8.0
- xgboost 3.2.0
- lightgbm 4.6.0
- imbalanced-learn 0.14.1
- shap 0.50.0
- numpy 2.3.5
- pandas 3.0.0
- matplotlib 3.10.0
- seaborn 0.13.2

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 快速开始 | Quick Start
1.Clone the repository
```bash
git clone https://github.com/2663638191/suicidal-intent-prediction-smote-enc-shap.git
cd suicidal-intent-prediction-smote-enc-shap
```
2.Install dependencies
```bash
pip install -r requirements.txt
```
3.Run the code
Jupyter Notebook: Open suicidal_intent.ipynb and run all cells
Python script:
```bash
python suicidal_intent.py
```

---

##  仓库结构 | Repository Structure
```plaintext
suicidal-intent-prediction-smote-enc-shap/
├── data/                     # 数据目录 | data directory
│   ├── sample_data.csv       # 示例数据格式（无真实数据） | Sample data format (no real data)
│   └── CCSMHSS_Scale_Description.xlsx  # CCSMHSS量表完整说明 | Complete Explanation of the CCSMHSS Scale
├── figures/                  # 论文结果图 | Graph of the research results
│   ├── ROC_Curves.png
│   ├── Recall_Comparison.png
│   ├── SHAP_Beeswarm.png
│   └── SHAP_Waterfall_179.png
├── .gitignore                # Git忽略规则 | Git Ignore Rules
├── LICENSE                   # MIT许可证 | MIT License
├── README.md                 # 本说明文档 | This instruction manual
├── requirements.txt          # 依赖包列表 | Dependency package list
├── suicidal_intent.ipynb     # Jupyter Notebook完整代码 | complete code
└── suicidal_intent.py        # Python脚本完整代码 | complete code
```
