"""心理筛查实验的算法与统计核心。所有拟合操作均由调用者限定在相应训练子集。"""
from __future__ import annotations

import itertools
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             confusion_matrix, brier_score_loss)
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

种子 = 42
模型名称 = {"LR": "逻辑回归", "DT": "决策树", "RF": "随机森林",
          "ET": "极端随机树", "XGB": "XGBoost", "LGBM": "LightGBM"}
策略名称 = ["未采样", "SMOTE-ENC"]
连续变量 = ["幻觉妄想", "焦虑", "抑郁", "偏执", "自卑", "敏感", "社交恐惧", "躯体化",
        "依赖", "敌对攻击", "冲动", "强迫", "网络成瘾", "自伤行为", "进食问题", "睡眠困扰",
        "学校适应困难", "人际关系困扰", "学业压力", "就业问题", "恋爱困扰"]
分类变量 = ["性别", "民族", "生源地", "是否独生"]
字段映射 = {"幻觉、妄想症状": "幻觉妄想", "就业压力": "就业问题"}
禁止特征 = ["自杀意念", "建议关注方向", "可能问题", "家庭关系满意度", "学校及专业满意度", "入学年份",
        "来源文件", "来源工作表", "源行号", "记录编号", "标签", "子集"]
指标名称 = {"AUC": "ROC-AUC", "AP": "平均精确率AP", "Recall": "召回率", "Specificity": "特异度",
          "PPV": "阳性预测值", "NPV": "阴性预测值", "F1": "F1值", "Accuracy": "准确率",
          "BalancedAccuracy": "平衡准确率", "Brier": "Brier分数", "FlagRate": "筛查阳性比例",
          "Review1000": "每千条记录复核数", "Miss1000": "每千条记录漏识标签阳性数",
          "ReviewPerPositive": "每识别一例标签阳性的复核数"}


class MixedPreprocessor:
    """连续变量用训练中位数插补并标准化；分类缺失预先统一为未知。"""
    def __init__(self, numeric=None):
        self.numeric = list(连续变量 if numeric is None else numeric)
        self.categorical = list(分类变量)

    def fit(self, X):
        assert not set(self.numeric + self.categorical) & set(禁止特征)
        self.medians_ = X[self.numeric].median()
        if self.medians_.isna().any():
            raise ValueError("训练部分存在整列缺失，不能插补为零：" + str(self.medians_[self.medians_.isna()].index.tolist()))
        self.scaler_ = StandardScaler().fit(X[self.numeric].fillna(self.medians_))
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float64)
        self.encoder_.fit(X[self.categorical].astype(str))
        self.names_ = self.numeric + list(self.encoder_.get_feature_names_out(self.categorical))
        self.groups_ = list(range(len(self.numeric)))
        for i, levels in enumerate(self.encoder_.categories_):
            self.groups_.extend([len(self.numeric) + i] * len(levels))
        self.groups_ = np.array(self.groups_)
        self.raw_names_ = self.numeric + self.categorical
        return self

    def mixed(self, X):
        n = self.scaler_.transform(X[self.numeric].fillna(self.medians_))
        c = X[self.categorical].astype(str).to_numpy()
        return n, c

    def design(self, n, c):
        # 以原始类别名构造 DataFrame，保证训练与应用的列顺序完全相同。
        cat = pd.DataFrame(c, columns=self.categorical)
        return np.hstack([n, self.encoder_.transform(cat)])

    def transform(self, X):
        return self.design(*self.mixed(X))

    def unknown_counts(self, X):
        return {c: int((~X[c].isin(v)).sum()) for c, v in zip(self.categorical, self.encoder_.categories_)}


class SMOTEENC:
    """SMOTE-ENC，而非 imblearn.SMOTENC。

    依照 Mukherjee & Khushi (2021), DOI:10.3390/asi4010018 及作者代码：
    ENC=(少数类观察数-类别总数*少数类占比)/(类别总数*少数类占比)，
    再乘以少数类连续变量总体标准差(ddof=0)的中位数。
    在编码距离空间中寻找少数类近邻、连续值插值；分类值直接取原类别
    的近邻众数，避免作者演示代码中不同类别编码碰撞后反查丢失类别的问题。
    众数并列时用固定随机种子抽签；不将编码作为最终模型的分类变量。
    """
    def __init__(self, ratio=0.8, k=5, seed=种子):
        self.ratio, self.k, self.seed = ratio, k, seed

    def fit_resample(self, n, c, y):
        y = np.asarray(y, dtype=int)
        n = np.asarray(n, dtype=float)
        c = np.asarray(c, dtype=str)
        counts = np.bincount(y, minlength=2)
        if min(counts) == 0 or counts[1] >= counts[0]:
            raise ValueError("本研究要求标签1为少数类，且训练子集中两类均存在。")
        self.original_counts_ = counts.copy()
        self.encoding_ = []
        prevalence = counts[1] / counts.sum()
        self.scale_ = float(np.median(n[y == 1].std(axis=0, ddof=0)))
        enc = np.zeros(c.shape, dtype=float)
        for j in range(c.shape[1]):
            mapping = {}
            for level in np.unique(c[:, j]):
                loc = c[:, j] == level
                expected = loc.sum() * prevalence
                mapping[level] = float(((y[loc] == 1).sum() - expected) / expected * self.scale_)
                enc[loc, j] = mapping[level]
            self.encoding_.append(mapping)
        wanted = int(np.floor(self.ratio * counts[0]))
        additional = wanted - counts[1]
        if additional < 0:
            raise ValueError("采样比例低于原始少数类/多数类比例，不可作为过采样候选。")
        if additional == 0:
            return n.copy(), c.copy(), y.copy()
        if counts[1] <= self.k:
            raise ValueError("少数类数量不足以采用指定近邻数。")
        minority_n, minority_c = n[y == 1], c[y == 1]
        space = np.hstack([minority_n, enc[y == 1]])
        neighbors = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=1).fit(space)
        # 显式去掉自身索引，重复向量时不能默认第一列一定是自身。
        indices = neighbors.kneighbors(space, return_distance=False)
        indices = np.vstack([row[row != i][:self.k] for i, row in enumerate(indices)])
        rng = np.random.default_rng(self.seed)
        flat = rng.integers(len(indices) * self.k, size=additional)
        rows, cols = np.divmod(flat, self.k)
        steps = rng.random((additional, 1))
        synthetic_n = minority_n[rows] + steps * (minority_n[indices[rows, cols]] - minority_n[rows])
        synthetic_c = np.empty((additional, c.shape[1]), dtype=object)
        for i, row in enumerate(rows):
            for j in range(c.shape[1]):
                levels, freq = np.unique(minority_c[indices[row], j], return_counts=True)
                synthetic_c[i, j] = rng.choice(levels[freq == freq.max()])
        self.synthetic_count_ = additional
        self.collisions_ = [len(m) - len(set(m.values())) for m in self.encoding_]
        for j in range(c.shape[1]):
            if not set(synthetic_c[:, j]).issubset(set(c[:, j])):
                raise AssertionError("合成分类值不合法")
        return (np.vstack([n, synthetic_n]), np.vstack([c, synthetic_c]).astype(str),
                np.r_[y, np.ones(additional, dtype=int)])


def make_model(code, p, seed=种子):
    p = dict(p)
    if code == "LR":
        # scikit-learn>=1.9 使用 l1_ratio 表达正则；旧版兼容写法在此分支内。
        from packaging.version import Version
        import sklearn
        lr = float(p.pop("l1_ratio", 0.0))
        opts = dict(C=p.pop("C", 1.0), solver="lbfgs" if lr == 0 else "saga",
                    max_iter=10000, tol=1e-5, random_state=seed)
        if Version(sklearn.__version__) >= Version("1.9"):
            opts["l1_ratio"] = lr
        else:
            opts["penalty"] = "l2" if lr == 0 else "elasticnet"
            if lr != 0:
                opts["l1_ratio"] = lr
        return LogisticRegression(**opts, **p)
    if code == "DT":
        return DecisionTreeClassifier(random_state=seed, **p)
    if code == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if code == "ET":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **p)
    if code == "XGB":
        return XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss",
                             objective="binary:logistic", verbosity=0, **p)
    if code == "LGBM":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1, deterministic=True,
                              force_col_wise=True, **p)
    raise KeyError(code)


def converged_fit(model, X, y):
    """以模型自身迭代数判断收敛，避免并行线程共享warnings状态导致误判。"""
    model.fit(X, y)
    history = []
    if isinstance(model, LogisticRegression):
        history.append({"预算": model.max_iter, "实际迭代": int(np.max(model.n_iter_))})
        for maximum in [50000, 200000]:
            if int(np.max(model.n_iter_)) < model.max_iter:
                break
            model.set_params(max_iter=maximum, warm_start=True)
            model.fit(X, y)
            history.append({"预算": maximum, "实际迭代": int(np.max(model.n_iter_))})
        if int(np.max(model.n_iter_)) >= model.max_iter:
            raise RuntimeError("逻辑回归增加迭代预算后仍未收敛，不得使用此拟合结果。")
    model.research_fit_history_ = history
    return model


def parameter_stages(code):
    if code == "LR":
        return {"C": 1.0, "l1_ratio": 0.0}, [
            {"C": [.01, .03, .1, .3, 1., 3., 10.], "l1_ratio": [0.]},
            {"C": [.03, .3, 3.], "l1_ratio": [.25, .5, .75]}]
    if code == "DT":
        return {"max_depth": 5, "min_samples_leaf": 5}, [
            {"max_depth": [3, 4, 5, 7, 10], "min_samples_leaf": [3, 8, 15]},
            {"criterion": ["gini", "entropy"], "min_samples_split": [2, 10, 30]}]
    if code in ("RF", "ET"):
        return {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 3}, [
            {"max_depth": [None, 5, 10, 15], "min_samples_leaf": [1, 3, 8]},
            {"n_estimators": [150, 300], "max_features": ["sqrt", .7, 1.]}]
    if code == "XGB":
        return {"n_estimators": 150, "learning_rate": .07, "max_depth": 3, "min_child_weight": 5}, [
            {"max_depth": [2, 3, 4, 5], "min_child_weight": [1, 5, 10]},
            {"learning_rate": [.03, .07, .12], "n_estimators": [150, 300]},
            {"subsample": [.7, 1.], "reg_lambda": [1., 5., 15.]}]
    if code == "LGBM":
        return {"n_estimators": 150, "learning_rate": .07, "num_leaves": 15, "min_child_samples": 20}, [
            {"num_leaves": [7, 15, 31, 63], "min_child_samples": [15, 30, 60]},
            {"learning_rate": [.03, .07, .12], "n_estimators": [150, 300]},
            {"colsample_bytree": [.7, 1.], "reg_lambda": [0., 5., 15.]}]
    raise KeyError(code)


def grid(d):
    return [dict(zip(d, values)) for values in itertools.product(*d.values())]


def raw_score(model, X, code):
    if code == "LR":
        return np.asarray(model.decision_function(X))
    return special.logit(np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))


@dataclass
class SigmoidCalibrator:
    intercept: float = 0.
    slope: float = 1.

    def fit(self, score, y):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)
        def f(ab):
            z = ab[0] + ab[1] * score
            loss = np.mean(np.logaddexp(0, z) - y * z)
            err = special.expit(z) - y
            return loss, np.array([err.mean(), np.mean(err * score)])
        res = optimize.minimize(f, [0., 1.], jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 3000})
        if not res.success and np.max(np.abs(res.jac)) > 1e-5:
            raise RuntimeError("Sigmoid校准未收敛：" + str(res.message))
        self.intercept, self.slope = map(float, res.x)
        self.gradient_ = float(np.max(np.abs(res.jac)))
        if not np.isfinite(res.x).all():
            raise RuntimeError("校准参数不是有限值。")
        return self

    def decision_function(self, score):
        return self.intercept + self.slope * np.asarray(score)

    def predict(self, score):
        return special.expit(self.decision_function(score))


def select_threshold(y, p, target=.88):
    """只接收开发集标签和概率；最高可行阈值保留全部同分记录。"""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("开发概率必须为[0,1]内有限数")
    positive = np.sort(p[y == 1])[::-1]
    needed = int(np.ceil(target * len(positive) - 1e-12))
    if needed < 1:
        raise ValueError("开发集需有阳性记录。")
    threshold = float(positive[needed - 1])
    actual = float(np.mean(p[y == 1] >= threshold))
    assert actual + 1e-12 >= target
    return threshold


@dataclass
class FittedExperiment:
    code: str
    strategy: str
    params: dict
    ratio: float = .8
    k: int = 5
    numeric: list | None = None
    seed: int = 种子

    def fit(self, X, y, devX, devy):
        self.preprocessor = MixedPreprocessor(self.numeric).fit(X)
        n, c = self.preprocessor.mixed(X)
        self.sampler = None
        if self.strategy == "SMOTE-ENC":
            self.sampler = SMOTEENC(self.ratio, self.k, self.seed)
            n, c, yy = self.sampler.fit_resample(n, c, y)
        else:
            yy = np.asarray(y)
        self.model = make_model(self.code, self.params, self.seed)
        converged_fit(self.model, self.preprocessor.design(n, c), yy)
        self.fit_history_ = self.model.research_fit_history_
        ds = self.base_score(devX)
        self.calibrator = SigmoidCalibrator().fit(ds, devy)
        self.threshold = select_threshold(devy, self.calibrator.predict(ds))
        self.train_prevalence = float(np.mean(y))
        self.sample_counts_ = {"拟合前": np.bincount(y, minlength=2).tolist(),
                               "拟合后": np.bincount(yy, minlength=2).tolist()}
        return self

    def base_score(self, X):
        return raw_score(self.model, self.preprocessor.transform(X), self.code)

    def decision_function(self, X):
        return self.calibrator.decision_function(self.base_score(X))

    def predict_proba(self, X):
        return special.expit(self.decision_function(X))


def metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else np.nan
    recall, spec = div(tp, tp + fn), div(tn, tn + fp)
    return {"AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
            "Recall": recall, "Specificity": spec, "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn),
            "F1": div(2 * tp, 2 * tp + fp + fn), "Accuracy": div(tp + tn, len(y)),
            "BalancedAccuracy": (recall + spec) / 2, "Brier": brier_score_loss(y, p),
            "FlagRate": div(tp + fp, len(y)), "Review1000": div((tp + fp) * 1000, len(y)),
            "Miss1000": div(fn * 1000, len(y)), "ReviewPerPositive": div(tp + fp, tp),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "Threshold": threshold}


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    out = np.empty(len(pvalues))
    out[order] = np.minimum(1., np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    return out


def delong_compare(y, probabilities, names, reference):
    """DeLong结构分量法；相同预测分值贡献0.5，不使用错误的非平均秩。"""
    y = np.asarray(y)
    pp = np.asarray(probabilities)
    pos, neg = pp[:, y == 1], pp[:, y == 0]
    vpos, vneg = [], []
    for a, b in zip(pos, neg):
        kernel = (a[:, None] > b[None, :]).astype(float) + .5 * (a[:, None] == b[None, :])
        vpos.append(kernel.mean(axis=1))
        vneg.append(kernel.mean(axis=0))
    vpos, vneg = np.asarray(vpos), np.asarray(vneg)
    aucs = vpos.mean(axis=1)
    covariance = np.cov(vpos, ddof=1) / pos.shape[1] + np.cov(vneg, ddof=1) / neg.shape[1]
    rows = []
    r = names.index(reference)
    for j, name in enumerate(names):
        if j == r:
            continue
        difference = float(aucs[r] - aucs[j])
        variance = max(0., float(covariance[r, r] + covariance[j, j] - 2 * covariance[r, j]))
        se = np.sqrt(variance)
        p = float(2 * stats.norm.sf(abs(difference) / se)) if se > 1e-15 else (1. if abs(difference) < 1e-12 else 0.)
        rows.append({"参照模型": reference, "对比模型": name, "AUC差值": difference,
                     "差值标准误": se, "差值95%下限": difference - 1.96 * se,
                     "差值95%上限": difference + 1.96 * se, "原始P值": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["Holm校正P值"] = holm(df["原始P值"])
    return df


def bootstrap_indices(y, rng):
    positive, negative = np.flatnonzero(np.asarray(y) == 1), np.flatnonzero(np.asarray(y) == 0)
    return np.r_[rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]


def fast_metric_array(y, probs, thresholds):
    """返回 模型×指标；所有模型共用同一个重采样索引。"""
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs)
    pred = probs >= np.asarray(thresholds)[:, None]
    pos = y == 1
    tp, fn = pred[:, pos].sum(axis=1), (~pred[:, pos]).sum(axis=1)
    tn, fp = (~pred[:, ~pos]).sum(axis=1), pred[:, ~pos].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, spec = tp / (tp + fn), tn / (tn + fp)
        values = {"AUC": np.array([roc_auc_score(y, p) for p in probs]),
                  "AP": np.array([average_precision_score(y, p) for p in probs]),
                  "Recall": recall, "Specificity": spec, "PPV": tp / (tp + fp), "NPV": tn / (tn + fn),
                  "F1": 2 * tp / (2 * tp + fp + fn), "Accuracy": (tp + tn) / len(y),
                  "BalancedAccuracy": (recall + spec) / 2, "Brier": np.mean((probs - y) ** 2, axis=1),
                  "FlagRate": (tp + fp) / len(y), "Review1000": 1000 * (tp + fp) / len(y),
                  "Miss1000": 1000 * fn / len(y), "ReviewPerPositive": (tp + fp) / tp}
    result = np.column_stack([values[k] for k in 指标名称])
    result[~np.isfinite(result)] = np.nan
    return result


def calibration_bins(y, p, bins=10):
    df = pd.DataFrame({"标签": np.asarray(y), "预测概率": np.asarray(p)})
    # 重复分位点合并；常数概率退化为单组。
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        df["分组"] = 0
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        df["分组"] = pd.cut(df["预测概率"], edges, include_lowest=True, labels=False)
    out = []
    for g, a in df.groupby("分组"):
        n, k = len(a), a["标签"].sum()
        rate = k / n
        # Wilson二项区间，避免小组中观察比例0或1时给出零宽区间。
        z = stats.norm.ppf(.975)
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        out.append({"分组": int(g) + 1, "人数": n, "阳性人数": int(k), "平均预测概率": a["预测概率"].mean(),
                    "观察阳性比例": rate, "观察比例95%下限": center - half, "观察比例95%上限": center + half})
    return pd.DataFrame(out)


def calibration_diagnostics(y, p):
    c = SigmoidCalibrator().fit(special.logit(np.clip(p, 1e-6, 1 - 1e-6)), y)
    return {"校准截距": c.intercept, "校准斜率": c.slope, "Brier分数": brier_score_loss(y, p)}


def aggregate_shap(values, groups, n_raw):
    out = np.zeros((len(values), n_raw))
    for j in range(n_raw):
        out[:, j] = values[:, np.asarray(groups) == j].sum(axis=1)
    return out


def explain_calibrated(fit, X, background, seed=种子, permutations=8):
    """解释最终校准后log-odds；返回原始变量层面的可加归因。

    LR使用背景经验分布的精确线性SHAP。其他模型对完整校准映射使用
    shap.PermutationExplainer，故概率截断、校准也包含在解释对象之内。
    非线性模型按原始变量整体遮蔽；分类值临时整数映射只用于传递合法类别。
    """
    p = fit.preprocessor
    B, A = p.transform(background), p.transform(X)
    if fit.code == "LR":
        coefficient = fit.calibrator.slope * fit.model.coef_[0]
        intercept = fit.calibrator.intercept + fit.calibrator.slope * fit.model.intercept_[0]
        phi = (A - B.mean(axis=0)) * coefficient
        base = np.full(len(A), intercept + B.mean(axis=0) @ coefficient)
        method = "经验背景分布下精确线性SHAP；分类独热贡献按原始变量求和"
    else:
        import shap
        names = p.raw_names_
        levels = {c: sorted(set(background[c].astype(str)) | set(X[c].astype(str))) for c in p.categorical}
        def pack(frame):
            z = frame[names].copy()
            for c in p.categorical:
                z[c] = pd.Categorical(z[c].astype(str), categories=levels[c]).codes
            return z.to_numpy(dtype=float)
        def mapping(z):
            z = pd.DataFrame(np.asarray(z), columns=names)
            for c in p.categorical:
                z[c] = np.asarray(levels[c], dtype=object)[z[c].to_numpy(dtype=int)]
            return fit.decision_function(z)
        masker = shap.maskers.Independent(pack(background), max_samples=len(background))
        explainer = shap.PermutationExplainer(mapping, masker, seed=seed)
        e = explainer(pack(X), max_evals=(2 * len(names) + 1) * permutations, silent=True)
        phi, base = np.asarray(e.values), np.asarray(e.base_values)
        method = f"原始变量整体遮蔽的完整校准映射Permutation SHAP，{permutations}个前后向排列对"
    value = aggregate_shap(phi, p.groups_, len(p.raw_names_)) if fit.code == "LR" else phi
    residual = fit.decision_function(X) - (np.asarray(base) + value.sum(axis=1))
    maximum = float(np.max(np.abs(residual)))
    if maximum > 1e-5:
        raise AssertionError(f"SHAP可加性检验失败，最大残差={maximum}")
    return value, np.asarray(base), {"算法": method, "解释尺度": "校准后log-odds", "最大可加性残差": maximum,
                                   "背景人数": len(background), "解释人数": len(X)}
