import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
warnings.filterwarnings("ignore")

FEATURE_EN_MAP = {
    "抑郁": "Depression",
    "焦虑": "Anxiety",
    "自伤行为": "Self-harm Behavior",
    "睡眠困扰": "Sleep Disturbance",
    "敏感": "Sensitivity",
    "自卑": "Inferiority",
    "冲动": "Impulsivity",
    "强迫": "Obsessive-compulsive",
    "社交恐惧": "Social Phobia",
    "敌对攻击": "Hostility",
    "躯体化": "Somatization",
    "学校适应困难": "School Adjustment",
    "就业压力": "Employment Stress",
    "学业压力": "Academic Stress",
    "恋爱困扰": "Relationship Stress",
    "人际关系困扰": "Interpersonal Stress",
    "进食问题": "Eating Disorder",
    "偏执": "Paranoia",
    "幻觉、妄想症状": "Hallucination-paranoia",
    "网络成瘾": "Internet Addiction",
    "依赖": "Dependence",
    "性别_男": "Gender_Male",
    "民族_汉族": "Ethnicity_Han",
    "民族_少数民族": "Ethnicity_Minority",
    "生源地_农村": "Origin_Rural",
    "是否独生_是": "Only_Child_Yes"
}

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class SuicideIntentionPredictor:
    def __init__(self, output_dir=None):
        if output_dir is None:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            self.output_dir = os.path.join(desktop, "result")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train_raw = None
        self.y_test_raw = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.feature_names = None
        self.cat_indices = None
        self.trained_models_raw = {}
        self.trained_models_smote = {}
        self.model_thresholds = {}
        self.metrics_raw = {}
        self.metrics_smote = {}

    def preprocess_data(self, file_paths):
        all_dfs = []
        target_sheets = ["严重心理危机", "一般心理问题", "潜在心理困扰", "无心理困扰"]

        for fp in file_paths:
            if os.path.exists(fp):
                df_dict = pd.read_excel(fp, sheet_name=target_sheets)
                all_dfs.extend(df_dict.values())
        
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df["一致性得分"] = pd.to_numeric(merged_df["一致性得分"], errors="coerce")
        cleaned_df = merged_df[merged_df["一致性得分"] >= 2].reset_index(drop=True)

        cleaned_df["Suicide_Ideation"] = cleaned_df["可能问题"].str.contains("自杀意图", na=False).astype(int)
        pos_count = cleaned_df["Suicide_Ideation"].sum()

        skip_cols = [
            "学历层次", "入学年份", "出生日期", "总分", "家庭关系满意度",
            "学校及专业满意度", "可能问题", "测评等级", "提交时间",
            "一致性得分", "总用时(秒)", "自杀意图(指标标准分)"
        ]
        skip_cols += [col for col in cleaned_df.columns if "(指标总分)" in col]
        cat_cols = ["性别", "民族", "生源地", "是否独生"]

        num_cols_original = [col for col in cleaned_df.columns if col not in skip_cols + cat_cols + ["Suicide_Ideation"]]
        num_cols_new = [col.replace("(指标标准分)", "").strip() for col in num_cols_original]
        cleaned_df.rename(columns=dict(zip(num_cols_original, num_cols_new)), inplace=True)

        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        cat_encoded = encoder.fit_transform(cleaned_df[cat_cols])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))
        cat_encoded_df = cat_encoded_df.drop(columns=[col for col in cat_encoded_df.columns if col.endswith("_nan")], errors='ignore')

        X = pd.concat([cleaned_df[num_cols_new], cat_encoded_df], axis=1)
        X = X.rename(columns=FEATURE_EN_MAP)
        y = cleaned_df["Suicide_Ideation"].values
        self.feature_names = X.columns.tolist()
        self.cat_indices = [X.columns.get_loc(col) for col in cat_encoded_df.rename(columns=FEATURE_EN_MAP).columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train_raw = y_train
        self.y_test_raw = y_test
        return X_train, X_test, y_train, y_test

    def apply_smote_enc(self):
        smote = SMOTENC(categorical_features=self.cat_indices, random_state=42, sampling_strategy=1.0)
        X_res, y_res = smote.fit_resample(self.X_train_raw, self.y_train_raw)
        self.X_train_smote = X_res
        self.y_train_smote = y_res
        return X_res, y_res

    def get_tuned_models(self):
        model_configs = {
            "Decision Tree": {
                "model": DecisionTreeClassifier(
                    random_state=42,
                    class_weight={0: 1, 1: 3},
                    criterion='entropy',
                    max_depth=3,
                    min_samples_leaf=1,
                    min_samples_split=2,
                    splitter='random'
                ),
                "threshold": 0.35
            },
            "Random Forest": {
                "model": RandomForestClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=3,
                    max_features='log2',
                    min_samples_leaf=3,
                    min_samples_split=2,
                    class_weight={0: 1, 1: 3},
                    n_jobs=-1
                ),
                "threshold": 0.4
            },
            "Extra Trees": {
                "model": ExtraTreesClassifier(
                    random_state=42,
                    n_estimators=400,
                    max_depth=4,
                    max_features='log2',
                    min_samples_leaf=1,
                    min_samples_split=2,
                    class_weight={0: 1, 1: 2},
                    n_jobs=-1
                ),
                "threshold": 0.45
            },
            "Logistic Regression": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(
                        random_state=42,
                        C=0.005,
                        class_weight={0: 1, 1: 6},
                        l1_ratio=0.3,
                        max_iter=2000,
                        penalty='elasticnet',
                        solver='saga'
                    ))
                ]),
                "threshold": 0.5
            }
        }
        if XGB_AVAILABLE:
            model_configs["XGBoost"] = {
                "model": xgb.XGBClassifier(
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1,
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=4,
                    min_child_weight=1,
                    scale_pos_weight=7.8,
                    gamma=0.05,
                    reg_alpha=0.05,
                    reg_lambda=1.2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective='binary:logistic'
                ),
                "threshold": 0.05
            }
        if LGB_AVAILABLE:
            model_configs["LightGBM"] = {
                "model": lgb.LGBMClassifier(
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    n_estimators=500,
                    learning_rate=0.04,
                    max_depth=4,
                    min_child_samples=20,
                    num_leaves=15,
                    scale_pos_weight=9.540487804878051,
                    reg_alpha=0.05,
                    reg_lambda=1.2,
                    subsample=0.85,
                    colsample_bytree=0.85
                ),
                "threshold": 0.05
            }
        return model_configs

    def calculate_core_metrics(self, model, X_test, y_test, threshold):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_proba >= threshold, 1, 0)
        return {
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_proba)
        }, y_proba

    def train_all_models(self):
        model_configs = self.get_tuned_models()
        for name, config in model_configs.items():
            model = config["model"]
            model.fit(self.X_train_raw, self.y_train_raw)
            metrics, _ = self.calculate_core_metrics(model, self.X_test_raw, self.y_test_raw, config["threshold"])
            self.trained_models_raw[name] = model
            self.metrics_raw[name] = metrics

        for name, config in model_configs.items():
            model = config["model"]
            model.fit(self.X_train_smote, self.y_train_smote)
            metrics, y_proba = self.calculate_core_metrics(model, self.X_test_raw, self.y_test_raw, config["threshold"])
            self.trained_models_smote[name] = model
            self.metrics_smote[name] = metrics
            setattr(self, f"{name}_proba", y_proba)

    def generate_comparison_table(self):
        model_names = list(self.metrics_raw.keys())
        table_data = []
        for model in model_names:
            row = {
                "Model": model,
                "Recall Before": round(self.metrics_raw[model]["Recall"], 4),
                "AUC-ROC Before": round(self.metrics_raw[model]["AUC-ROC"], 4),
                "Recall After": round(self.metrics_smote[model]["Recall"], 4),
                "AUC-ROC After": round(self.metrics_smote[model]["AUC-ROC"], 4)
            }
            table_data.append(row)
        
        en_df = pd.DataFrame(table_data).set_index("Model").T
        en_path = os.path.join(self.output_dir, "Model_Comparison.xlsx")
        en_df.to_excel(en_path, engine='openpyxl')

    def plot_recall(self):
        model_names = list(self.metrics_smote.keys())
        recall_values = [self.metrics_smote[name]["Recall"] for name in model_names]

        plt.figure(figsize=(10,6))
        plt.bar(model_names, recall_values, color=plt.cm.Set3(np.linspace(0,1,len(model_names))), edgecolor='black')
        for i, v in enumerate(recall_values): plt.text(i, v+0.01, f"{v:.4f}", ha='center')
        plt.title("Recall Comparison After SMOTE Augmentation", fontweight='bold', fontsize=16)
        plt.ylabel("Recall Score", fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=12)
        plt.ylim(0,1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "Recall_Comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc(self):
        y_test = self.y_test_raw
        model_names = list(self.metrics_smote.keys())

        plt.figure(figsize=(10,8))
        for name in model_names:
            fpr, tpr, _ = roc_curve(y_test, getattr(self, f"{name}_proba"))
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
        plt.plot([0,1],[0,1], 'k--', label="Random Classifier")
        plt.xlabel("False Positive Rate (FPR)", fontsize=14)
        plt.ylabel("True Positive Rate (TPR)", fontsize=14)
        plt.title("ROC Curves After SMOTE Augmentation", fontweight='bold', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ROC_Curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def shap_analysis(self):
        best_model = self.trained_models_smote["Logistic Regression"]
        sample_list = [179, 362, 260, 559, 752, 7, 42, 88, 98, 250, 460]

        scaler = best_model.named_steps["scaler"]
        X_test_shap = pd.DataFrame(scaler.transform(self.X_test_raw), columns=self.feature_names)
        X_train_shap = pd.DataFrame(scaler.transform(self.X_train_raw), columns=self.feature_names)
        explainer = shap.Explainer(best_model.named_steps["model"], X_train_shap)
        shap_values = explainer(X_test_shap)

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values, X_test_shap, show=False, max_display=11)
        plt.title("SHAP Beeswarm Plot (Global Feature Importance)", fontweight='bold', fontsize=16, pad=20)
        plt.xlabel("SHAP Value", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "SHAP_Beeswarm.png"), dpi=300, bbox_inches='tight')
        plt.close()

        for sample_id in sample_list:
            idx = sample_id - 1
            plt.figure(figsize=(14,8))
            shap.plots.waterfall(shap_values[idx], max_display=6, show=False)
            plt.title("SHAP Waterfall Plot", fontweight='bold', fontsize=16, pad=20)
            plt.xlabel("Feature Contribution", fontsize=14)
            plt.ylabel("Features", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"SHAP_Waterfall_{sample_id}.png"), dpi=300, bbox_inches='tight')
            plt.close()

        shap_df = pd.DataFrame(shap_values.values, columns=self.feature_names)
        shap_df.insert(0, "Sample_ID", range(1, len(shap_df)+1))
        shap_df.to_excel(os.path.join(self.output_dir, "SHAP_Values_Detail.xlsx"), index=False, engine='openpyxl')

    def run_full_pipeline(self, file_paths=None):
        if file_paths is None:
            file_paths = [
                r"C:\Users\DELL\Desktop\心理测评\测评结果-2025-1387.xlsx",
                r"C:\Users\DELL\Desktop\心理测评\测评结果-2024-1284.xlsx",
                r"C:\Users\DELL\Desktop\心理测评\测评结果-2023-1277.xlsx",
                r"C:\Users\DELL\Desktop\心理测评\测评结果-2022-1269.xlsx",
                r"C:\Users\DELL\Desktop\心理测评\测评结果-2021-1018.xlsx"
            ]

        valid_files = [fp for fp in file_paths if os.path.exists(fp)]
        if not valid_files:
            print("ERROR: Please update the file path!")
            return

        try:
            self.preprocess_data(valid_files)
            self.apply_smote_enc()
            self.train_all_models()
            self.generate_comparison_table()
            self.plot_recall()
            self.plot_roc()
            self.shap_analysis()

            print("\n✅ ALL TASKS COMPLETED!")
            print(f"Results saved to: {self.output_dir}")
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    predictor = SuicideIntentionPredictor()
    predictor.run_full_pipeline()