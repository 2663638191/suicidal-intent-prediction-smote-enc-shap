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
pip install -r requirements.txt```
