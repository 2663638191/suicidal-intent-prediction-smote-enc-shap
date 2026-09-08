# 不平衡数据下大学生自杀意图的可解释机器学习预测

本仓库提供基于 SMOTE-ENC 与 SHAP 的可解释机器学习预测框架的**可复现代码与结果**，用于识别大学生自杀意图，弥补自陈量表直接测查中因社会期许偏差与回避作答导致的效度局限。

> ⚠️ 数据说明：原始心理普查数据涉及隐私，依据相关规定不予公开。仓库仅提供**代码**与**匿名化汇总结果**，可复现分析逻辑或迁移至其他数据集。

## 方法概览

- **数据**：2021—2025 级大学新生《中国大学生心理健康筛查量表》（CCSMHSS）心理普查数据，最终纳入有效样本 3068 例（自杀意图阳性 527 例，占 17.18%）。
- **特征工程**：21 个维度标准分 + 4 个人口学变量（独热编码）。
- **不平衡处理**：SMOTE-ENC（仅训练集，平衡至 50%）。
- **模型**：逻辑回归、决策树、随机森林、极端随机树、XGBoost、LightGBM 六种，网格搜索 + 5 折分层交叉验证调参（以召回率为准则，阈值统一为 0.5）。
- **可解释性**：SHAP（蜂群图 + 瀑布图）。
- **模型比较与校准**：DeLong 检验、Brier 分数、Hosmer-Lemeshow 检验、校准曲线。

## 主要结果

| 模型 | 增强前召回率 | 增强后召回率 | AUC-ROC |
|---|---|---|---|
| 逻辑回归 | 0.810 | **0.981** | 0.846 |
| 决策树 | 0.639 | 0.899 | 0.790 |
| 随机森林 | 0.620 | 0.918 | 0.832 |
| 极端随机树 | 0.279 | 0.962 | 0.835 |
| XGBoost | 0.684 | 0.835 | 0.852 |
| LightGBM | 0.715 | 0.810 | 0.845 |

- 逻辑回归为最优模型：召回率 0.981（漏检率 1.9%），AUC-ROC 0.846，与集成模型差异无统计学意义（DeLong 检验）。
- SHAP 识别核心预测因子（按贡献排序）：幻觉妄想、抑郁、躯体化、自伤行为、敌对攻击；人口学变量贡献甚微。

##  仓库结构
```plaintext
suicidal-intent-prediction-smote-enc-shap/
├── README.md                  # 中文版
├── README_EN.md               # 英文版
├── LICENSE                    
├── .gitignore                 # 排除数据与中间产物
├── requirements.txt           # 需要准备的库
├── code/
│   └── main_pipeline.ipynb    # 代码
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
```
