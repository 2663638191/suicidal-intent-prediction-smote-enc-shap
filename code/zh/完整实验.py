"""按《实验流程最终版》执行五年合并数据的完整实验。运行说明见同目录README。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

代码目录 = Path(__file__).resolve().parent
if (代码目录 / "依赖环境").exists():
    sys.path.insert(0, str(代码目录 / "依赖环境"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(代码目录 / "运行缓存"))

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
import warnings
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from 算法与统计 import *


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=default, allow_nan=False)
    temp = path.with_suffix(path.suffix + ".临时")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def log(text):
    print(time.strftime("[%H:%M:%S] ") + text, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_data(folder, out):
    tables, audits, sources, category_issues = [], [], [], []
    must = set(连续变量 + 分类变量 + ["建议关注方向"])
    years = list(range(2021, 2026))
    for year in years:
        path = Path(folder) / f"{year}.xlsx"
        if not path.is_file():
            raise FileNotFoundError(f"缺少预定年份文件：{path}")
        sources.append({"文件": path.name, "字节数": path.stat().st_size, "SHA256": digest(path)})
        with pd.ExcelFile(path, engine="openpyxl") as book:
            sheets = {sheet: pd.read_excel(book, sheet_name=sheet) for sheet in book.sheet_names}
        for sheet, original in sheets.items():
            original.columns = [unicodedata.normalize("NFKC", str(c)).strip() for c in original.columns]
            d = original.rename(columns=字段映射).copy()
            if d.columns.duplicated().any():
                raise ValueError(f"字段映射后出现重复列：{path.name}/{sheet}")
            missing = must - set(d.columns)
            if missing:
                raise ValueError(f"缺少必要字段，不得将整列缺失当作全阴性：{path.name}/{sheet}/{missing}")
            # 只移除全部导出字段均为空的表格空行；不按分数相同删除记录。
            nonempty = d.notna().any(axis=1)
            empty_count = int((~nonempty).sum())
            d = d.loc[nonempty].copy()
            if len(d) == 0:
                audits.append({"文件": path.name, "工作表": sheet, "读取行数": len(original),
                               "排除全空行": empty_count, "有效行数": 0, "标签阳性数": 0,
                               "关注方向空白阴性数": 0})
                continue
            d["记录编号"] = [f"{year}_{sheet}_{i+2}" for i in d.index]
            d["来源文件"], d["来源工作表"], d["源行号"] = path.name, sheet, d.index + 2
            blank = d["建议关注方向"].isna() | d["建议关注方向"].astype(str).str.strip().eq("")
            d["标签"] = d["建议关注方向"].fillna("").astype(str).map(lambda x: unicodedata.normalize("NFKC", x).strip()).str.contains("自杀意念", regex=False).astype(int)
            for c in 连续变量:
                previous = d[c].copy()
                number = pd.to_numeric(previous, errors="coerce")
                invalid = (previous.notna() & number.isna()) | np.isinf(number)
                if invalid.any():
                    raise ValueError(f"{path.name}/{sheet}/{c}存在非法数值或重复表头，需核对而非静默插补")
                if number.notna().sum() == 0:
                    raise ValueError(f"{path.name}/{sheet}/{c}整列缺失")
                d[c] = number.replace([np.inf, -np.inf], np.nan)
            for c in 分类变量:
                s = d[c].astype("string").str.strip()
                unknown = s.isna() | s.isin(["", "待完善", "未知", "不详", "无数据", "nan", "None"])
                d[c] = s.mask(unknown, "未知").astype(str)
                if unknown.any():
                    category_issues.append({"文件": path.name, "工作表": sheet, "变量": c,
                                            "问题": "缺失或待完善统一为未知", "数量": int(unknown.sum())})
            audits.append({"文件": path.name, "工作表": sheet, "读取行数": len(original),
                           "排除全空行": empty_count, "有效行数": len(d),
                           "标签阳性数": int(d["标签"].sum()), "关注方向空白阴性数": int(blank.sum())})
            tables.append(d)
    d = pd.concat(tables, ignore_index=True)
    assert d["记录编号"].is_unique
    if d["标签"].nunique() != 2:
        raise ValueError("结局必须同时含阴性与阳性。")
    csv(pd.DataFrame(audits), out / "表格" / "数据读取与清洗记录.csv")
    csv(pd.DataFrame(sources), out / "复现记录" / "原始文件校验值.csv")
    csv(pd.DataFrame(category_issues, columns=["文件", "工作表", "变量", "问题", "数量"]),
        out / "表格" / "字段异常处理记录.csv")
    dictionary = [{"变量": c, "角色": "连续预测变量", "处理": "训练子集内中位数插补及标准化"} for c in 连续变量]
    dictionary += [{"变量": c, "角色": "分类预测变量", "处理": "缺失归未知；重采样后独热编码"} for c in 分类变量]
    dictionary += [{"变量": c, "角色": "禁止进入X", "处理": "不作为预测特征"} for c in 禁止特征]
    csv(pd.DataFrame(dictionary), out / "表格" / "变量字典与角色.csv")
    duplicate = d.duplicated(subset=连续变量 + 分类变量, keep=False)
    write_json(out / "复现记录" / "清洗规则与核查.json", {
        "关注方向空白含义": "有效的未提示问题记录，标签为0", "标签判定": "建议关注方向包含自杀意念为1；其余为0",
        "数值自杀意念得分用于建模": False, "一致性得分用于建模或过滤": False,
        "完全同特征记录数": int(duplicate.sum()), "同特征记录处理": "没有身份标识，不能认定同一受测者，保留",
        "系统标准分处理": "不重建常模；模型训练子集内z标准化仅服务数值计算",
        "年份用途": "仅文件定位和来源审计；五年合并后分层随机划分，不进入X"})
    return d, pd.DataFrame(audits), sources


def make_split(d, out):
    idx = np.arange(len(d))
    train, remainder = train_test_split(idx, test_size=.4, random_state=种子, stratify=d["标签"])
    dev, test = train_test_split(remainder, test_size=.5, random_state=种子, stratify=d.loc[remainder, "标签"])
    assert not (set(train) & set(dev) or set(train) & set(test) or set(test) & set(dev))
    assert len(set(train) | set(dev) | set(test)) == len(d)
    subset = pd.Series(index=d.index, dtype="str")
    for ids, name in [(train, "训练集"), (dev, "开发验证集"), (test, "最终测试集")]:
        subset.loc[ids] = name
    d["子集"] = subset
    csv(d[["记录编号", "来源文件", "来源工作表", "源行号", "标签", "子集"]], out / "私密记录_禁止公开" / "固定数据划分.csv")
    rows = []
    for name in ["合并数据", "训练集", "开发验证集", "最终测试集"]:
        a = d if name == "合并数据" else d[d["子集"] == name]
        rows.append({"子集": name, "总人数": len(a), "标签阳性数": int(a["标签"].sum()),
                     "标签阴性数": int((a["标签"] == 0).sum()), "阳性比例": a["标签"].mean()})
    csv(pd.DataFrame(rows), out / "表格" / "样本构成与固定划分.csv")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=种子).split(train, d.loc[train, "标签"]))
    fold_record = []
    for k, (tr, va) in enumerate(folds, 1):
        assert not set(tr) & set(va)
        for i in va:
            fold_record.append({"记录编号": d.loc[train[i], "记录编号"], "验证折": k})
    csv(pd.DataFrame(fold_record), out / "私密记录_禁止公开" / "训练集五折索引.csv")
    return train, dev, test, folds


def describe(d, train, out):
    numeric_rows, category_rows, missing_rows = [], [], []
    for c in 连续变量 + 分类变量:
        missing_rows.append({"变量": c, "缺失或未知数": int(d[c].isna().sum()) if c in 连续变量 else int(d[c].eq("未知").sum()),
                             "分母": len(d), "处理方式": "训练中位数插补" if c in 连续变量 else "未知类别保留"})
    for label, group in [("全体", d), ("标签阴性", d[d["标签"] == 0]), ("标签阳性", d[d["标签"] == 1])]:
        for c in 连续变量:
            s = group[c]
            numeric_rows.append({"分组": label, "变量": c, "非缺失数": int(s.count()), "均值": s.mean(),
                                 "标准差": s.std(), "中位数": s.median(), "四分位数下限": s.quantile(.25),
                                 "四分位数上限": s.quantile(.75), "最小值": s.min(), "最大值": s.max()})
        for c in 分类变量:
            for value, count in group[c].value_counts().items():
                category_rows.append({"分组": label, "变量": c, "类别": value, "人数": count,
                                      "分母": len(group), "比例": count / len(group)})
    csv(pd.DataFrame(numeric_rows), out / "表格" / "心理维度描述统计.csv")
    csv(pd.DataFrame(category_rows), out / "表格" / "人口学变量描述统计.csv")
    csv(pd.DataFrame(missing_rows), out / "表格" / "变量缺失与未知类别.csv")
    # 探索性关联只使用训练集，绝不据此单变量筛选X。
    a = d.loc[train]
    correlations = []
    for c in 连续变量:
        valid = a[c].notna()
        r, p = stats.pointbiserialr(a.loc[valid, "标签"], a.loc[valid, c])
        correlations.append({"变量": c, "有效训练人数": int(valid.sum()), "点二列相关系数": r, "原始P值": p})
    cor = pd.DataFrame(correlations)
    cor["Holm校正P值"] = holm(cor["原始P值"])
    csv(cor, out / "表格" / "训练集点二列相关与校正.csv")
    spearman = a[连续变量].corr(method="spearman")
    csv(spearman.reset_index(names="变量"), out / "表格" / "训练集Spearman相关矩阵.csv")
    imputed = a[连续变量].fillna(a[连续变量].median()).to_numpy(dtype=float)
    vif_rows = []
    for j, name in enumerate(连续变量):
        yy = imputed[:, j]
        rest = np.c_[np.ones(len(imputed)), np.delete(imputed, j, axis=1)]
        beta = np.linalg.lstsq(rest, yy, rcond=None)[0]
        sst = np.sum((yy - yy.mean()) ** 2)
        sse = np.sum((yy - rest @ beta) ** 2)
        vif = sst / sse if sse > 1e-12 and sst > 0 else np.nan
        vif_rows.append({"变量": name, "VIF": vif, "样本": "训练集中位数插补后"})
    csv(pd.DataFrame(vif_rows), out / "表格" / "训练集方差膨胀因子.csv")
    return spearman


class FoldCache:
    def __init__(self, X, y, folds, numeric):
        self.X, self.y, self.folds, self.numeric = X, np.asarray(y), folds, numeric
        self.prepared, self.resampled, self.audits = [], {}, []
        for tr, va in folds:
            p = MixedPreprocessor(numeric).fit(X.iloc[tr])
            n, c = p.mixed(X.iloc[tr])
            self.prepared.append((p, n, c, self.y[tr], p.transform(X.iloc[va]), self.y[va]))

    def get(self, strategy, ratio=.8, k=5):
        key = strategy, ratio if strategy != "未采样" else 0, k if strategy != "未采样" else 0
        if key not in self.resampled:
            result = []
            for i, (p, n, c, yy, validation, vy) in enumerate(self.prepared):
                if strategy == "SMOTE-ENC":
                    sampler = SMOTEENC(ratio, k, 种子 + i)
                    nn, cc, ry = sampler.fit_resample(n, c, yy)
                    self.audits.append({"训练折": i+1, "采样比例": ratio, "近邻数": k,
                        "采样前阴性数": int(np.sum(yy == 0)), "采样前阳性数": int(np.sum(yy == 1)),
                        "采样后阴性数": int(np.sum(ry == 0)), "采样后阳性数": int(np.sum(ry == 1)),
                        "类别合法": all(set(cc[:, j]).issubset(set(c[:, j])) for j in range(c.shape[1])),
                        "关联编码尺度": sampler.scale_, "碰撞分类数": sum(getattr(sampler, "collisions_", []))})
                else:
                    nn, cc, ry = n, c, yy
                result.append((p.design(nn, cc), ry, validation, vy))
            self.resampled[key] = result
        return self.resampled[key]


def evaluate_candidate(code, params, prepared):
    ap, auc, losses, iters, fits = [], [], [], [], []
    started = time.monotonic()
    for fold, (a, y, b, vy) in enumerate(prepared):
        model = make_model(code, params, 种子 + fold)
        converged_fit(model, a, y)
        p = model.predict_proba(b)[:, 1]
        ap.append(average_precision_score(vy, p))
        auc.append(roc_auc_score(vy, p))
        losses.append(log_loss(vy, p, labels=[0, 1]))
        iters.append(int(np.max(getattr(model, "n_iter_", [0]))))
        fits.append({"折": fold + 1, "AP": ap[-1], "AUC": auc[-1], "对数损失": losses[-1],
                     "最终迭代预算": int(getattr(model, "max_iter", 0)),
                     "达到迭代上限": bool(code == "LR" and iters[-1] >= model.max_iter),
                     "收敛过程": model.research_fit_history_})
    return {"平均AP": float(np.mean(ap)), "AP标准差": float(np.std(ap, ddof=1)),
            "平均AUC": float(np.mean(auc)), "平均对数损失": float(np.mean(losses)),
            "最大迭代数": max(iters), "耗时秒": time.monotonic() - started, "五折指标": fits}


def tune(code, strategy, cache, checkpoints, workers, prefix="主分析"):
    path = checkpoints / f"{prefix}_{模型名称[code]}_{strategy}_搜索.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    seen = {r["候选键"]: r for r in old}
    all_rows = list(old)
    base, stages = parameter_stages(code)
    ratio, k = .8, 5
    def run_batch(candidates, sampling, stage):
        prepared = cache.get(strategy, *sampling)
        keys, new = [], []
        for params in candidates:
            key = json.dumps([params, sampling], sort_keys=True, ensure_ascii=False)
            keys.append(key)
            if key not in seen:
                new.append((key, params))
        if len(all_rows) + len(new) > 60:
            raise RuntimeError("候选预算超过60，须在拟合前阻断")
        if new:
            def evaluate(item):
                key, params = item
                try:
                    r = dict(evaluate_candidate(code, params, prepared), 状态="成功")
                except Exception as exc:
                    r = {"状态": "失败", "错误": f"{type(exc).__name__}: {exc}"}
                return dict(r, 候选键=key, 参数=params, 采样比例=sampling[0] if strategy == "SMOTE-ENC" else None,
                            近邻数=sampling[1] if strategy == "SMOTE-ENC" else None, 阶段=stage)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(evaluate, new):
                    seen[result["候选键"]] = result
                    all_rows.append(result)
                    write_json(path, all_rows)
            log(f"{prefix}/{模型名称[code]}/{strategy}：{stage}完成，累计{len(all_rows)}个候选")
        result = [seen[key] for key in keys if seen[key].get("状态", "成功") == "成功"]
        if not result:
            raise RuntimeError(f"{code}/{strategy}/{stage}没有成功候选，见搜索日志")
        # 仅数值精度内AP相同才用对数损失破除并列。
        best_ap = max(x["平均AP"] for x in result)
        near = [x for x in result if x["平均AP"] >= best_ap - 1e-12]
        return min(near, key=lambda x: (x["平均对数损失"], json.dumps(x["参数"], sort_keys=True)))
    # 逐组搜索，每阶段最多同时改变三个参数；两种策略使用相同的分类器网格。
    for i, stage in enumerate(stages, 1):
        candidates = [dict(base, **delta) for delta in grid(stage)]
        winner = run_batch(candidates, (ratio, k), f"分类器第{i}阶段")
        base = winner["参数"]
    if strategy == "SMOTE-ENC":
        feasible = []
        for rr in [.4, .6, .8, 1.]:
            for kk in [3, 5, 7]:
                if all(np.sum(y == 1) > kk and rr * np.sum(y == 0) >= np.sum(y == 1)
                       for _, _, _, y, _, _ in cache.prepared):
                    feasible.append((rr, kk))
        sampler_winners = [run_batch([base], sample, "采样器阶段") for sample in feasible]
        best_ap = max(x["平均AP"] for x in sampler_winners)
        winner = min([x for x in sampler_winners if x["平均AP"] >= best_ap - 1e-12],
                     key=lambda x: (x["平均对数损失"], x["采样比例"], x["近邻数"]))
        ratio, k = winner["采样比例"], winner["近邻数"]
    # 预留末轮联合复核预算：分类器两个敏感参数，或一个分类器参数加采样比例、近邻数。
    critical = {"LR": ("C", "l1_ratio"), "DT": ("max_depth", "min_samples_leaf"),
                "RF": ("max_depth", "min_samples_leaf"), "ET": ("max_depth", "min_samples_leaf"),
                "XGB": ("max_depth", "min_child_weight"), "LGBM": ("num_leaves", "min_child_samples")}[code]
    def adjacent(name):
        values = []
        for stage in stages:
            values.extend(stage.get(name, []))
        values = list(dict.fromkeys(values + [base.get(name)]))
        current = base.get(name)
        values.sort(key=lambda v: (v != current, abs(float(v or 100) - float(current or 100))))
        return values[:3]
    if strategy == "未采样":
        candidates = [dict(base, **delta) for delta in grid({n: adjacent(n) for n in critical})]
        run_batch(candidates, (ratio, k), "末轮有限联合复核")
    else:
        samples = sorted(feasible, key=lambda s: (abs(s[0]-ratio), abs(s[1]-k)))[:4]
        candidates = [dict(base, **{critical[0]: v}) for v in adjacent(critical[0])]
        for sample in samples:
            run_batch(candidates, sample, "末轮有限联合复核")
    successful = [r for r in all_rows if r.get("状态", "成功") == "成功"]
    best_ap = max(r["平均AP"] for r in successful)
    winner = min([r for r in successful if r["平均AP"] >= best_ap-1e-12], key=lambda r: (r["平均对数损失"], r["候选键"]))
    base = winner["参数"]
    if strategy == "SMOTE-ENC":
        ratio, k = winner["采样比例"], winner["近邻数"]
    assert len(all_rows) <= 60
    return base, ratio, k, all_rows


def fit_comparison(d, train, dev, folds, numeric, codes, out, workers, prefix="主分析"):
    cache = FoldCache(d.iloc[train], d.loc[train, "标签"].to_numpy(), folds, numeric)
    fitted, searches, development = {}, [], []
    folder = out / "私密记录_禁止公开" / "模型与断点"
    folder.mkdir(parents=True, exist_ok=True)
    for code in codes:
        for strategy in 策略名称:
            key = f"{code}|{strategy}"
            model_path = folder / f"{prefix}_{模型名称[code]}_{strategy}.joblib"
            params, ratio, k, rows = tune(code, strategy, cache, folder, workers, prefix)
            for r in rows:
                row = dict(r)
                row["参数"] = json.dumps(row["参数"], ensure_ascii=False, sort_keys=True)
                row["五折指标"] = json.dumps(row.get("五折指标", []), ensure_ascii=False)
                row.pop("候选键", None)
                searches.append(dict(row, 分析=prefix, 模型=模型名称[code], 策略=strategy))
            if model_path.exists():
                fit = joblib.load(model_path)
            else:
                fit = FittedExperiment(code, strategy, params, ratio, k, numeric).fit(
                    d.iloc[train], d.loc[train, "标签"].to_numpy(), d.iloc[dev], d.loc[dev, "标签"].to_numpy())
                joblib.dump(fit, model_path, compress=3)
            p = fit.predict_proba(d.iloc[dev])
            values = metrics(d.loc[dev, "标签"], p, fit.threshold)
            development.append(dict(模型=模型名称[code], 模型代码=code, 策略=strategy, **values,
                                    校准截距=fit.calibrator.intercept, 校准斜率=fit.calibrator.slope,
                                    分类器参数=json.dumps(params, ensure_ascii=False, sort_keys=True),
                                    采样比例=ratio if strategy == "SMOTE-ENC" else np.nan,
                                    近邻数=k if strategy == "SMOTE-ENC" else np.nan,
                                    训练阳性数=fit.sample_counts_["拟合前"][1],
                                    重采样后阳性数=fit.sample_counts_["拟合后"][1],
                                    未见类别数=sum(fit.preprocessor.unknown_counts(d.iloc[dev]).values())))
            fitted[key] = fit
            log(f"{prefix}/{模型名称[code]}/{strategy}：训练及开发流程已冻结")
    csv(pd.DataFrame(searches), out / "复现记录" / f"{prefix}_全部候选与交叉验证.csv")
    csv(pd.DataFrame(cache.audits), out / "复现记录" / f"{prefix}_折内采样核对.csv")
    development = pd.DataFrame(development)
    csv(development.rename(columns=指标名称), out / "表格" / f"{prefix}_开发验证表现与参数.csv")
    return fitted, development


def freeze_selection(dev, out, signature):
    qualified = dev[dev["Recall"] >= .88 - 1e-12].copy()
    if not len(qualified):
        raise RuntimeError("开发阶段无达标候选，不自动改变预定筛查目标")
    best = qualified.sort_values(["Specificity", "AUC", "Brier"], ascending=[False, False, True]).iloc[0]
    close = qualified[(qualified["Specificity"] >= best["Specificity"] - .02) &
                      (qualified["AUC"] >= best["AUC"] - .02) & (qualified["Brier"] <= best["Brier"] + .01)]
    lr = close[close["模型代码"] == "LR"]
    pool = lr if len(lr) else close
    selected = pool.sort_values(["Specificity", "AUC", "Brier"], ascending=[False, False, True]).iloc[0]
    key = f"{selected['模型代码']}|{selected['策略']}"
    result = {"主模型键": key, "主模型": selected["模型"], "主策略": selected["策略"],
              "开发集召回率": float(selected["Recall"]), "开发集特异度": float(selected["Specificity"]),
              "开发集ROC-AUC": float(selected["AUC"]), "开发集Brier": float(selected["Brier"]),
              "候选近似范围": {"特异度差": .02, "AUC差": .02, "Brier增量": .01},
              "规则": "开发集敏感度合格后比较特异度；在预定近似范围内优先逻辑回归，未用最终测试集选模",
              "研究签名": signature, "冻结时间UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = out / "复现记录" / "测试集启封前的冻结方案.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["主模型键"] != key or previous["研究签名"] != signature:
            raise RuntimeError("已有冻结方案与当前研究不一致，需独立结果目录，禁止覆盖后重新选择。")
        result = previous
    else:
        write_json(path, result)
    log(f"冻结的主模型：{selected['模型']}/{selected['策略']}；现在进入测试集评价")
    return result


def test_evaluation(fitted, d, test, out, prefix="主分析"):
    rows, probs, private, diagnostics, bins = [], {}, [], [], []
    y = d.loc[test, "标签"].to_numpy()
    for key, fit in fitted.items():
        p = fit.predict_proba(d.iloc[test])
        raw = special.expit(fit.base_score(d.iloc[test]))
        values = metrics(y, p, fit.threshold)
        rows.append(dict(键=key, 模型=模型名称[fit.code], 策略=fit.strategy, **values,
                         未见类别数=sum(fit.preprocessor.unknown_counts(d.iloc[test]).values())))
        probs[key] = p
        for name, pp in [("校准前", raw), ("校准后", p)]:
            diagnostics.append(dict(模型=模型名称[fit.code], 策略=fit.strategy, 阶段=name,
                                    **calibration_diagnostics(y, pp)))
            group = calibration_bins(y, pp)
            group.insert(0, "阶段", name)
            group.insert(0, "策略", fit.strategy)
            group.insert(0, "模型", 模型名称[fit.code])
            bins.append(group)
        for rid, yy, prob, rp in zip(d.loc[test, "记录编号"], y, p, raw):
            private.append({"记录编号": rid, "系统标签": yy, "模型": 模型名称[fit.code], "策略": fit.strategy,
                            "校准前概率": rp, "校准后概率": prob, "固定阈值": fit.threshold,
                            "预测标签": int(prob >= fit.threshold)})
    table = pd.DataFrame(rows)
    csv(table.rename(columns=指标名称), out / "表格" / f"{prefix}_测试集完整指标.csv")
    csv(pd.DataFrame(private), out / "私密记录_禁止公开" / f"{prefix}_测试集逐记录预测.csv")
    csv(pd.DataFrame(diagnostics), out / "表格" / f"{prefix}_概率校准诊断.csv")
    csv(pd.concat(bins, ignore_index=True), out / "表格" / f"{prefix}_校准曲线分组与区间.csv")
    return table, probs, pd.concat(bins, ignore_index=True)


def paired_bootstrap(fitted, probs, y, table, out, repeats=2000):
    path = out / "私密记录_禁止公开" / "模型与断点" / "主分析_配对重采样分布.npz"
    keys, names = list(fitted), list(指标名称)
    if path.exists():
        with np.load(path) as old:
            distributions = old["指标分布"]
        if distributions.shape != (repeats, len(keys), len(names)):
            raise RuntimeError("重采样配置与已保存断点不一致。")
    else:
        rng = np.random.default_rng(20261022)
        distributions = np.empty((repeats, len(keys), len(names)))
        probabilities = np.asarray([probs[k] for k in keys])
        thresholds = np.array([fitted[k].threshold for k in keys])
        for b in range(repeats):
            ids = bootstrap_indices(y, rng)
            distributions[b] = fast_metric_array(y[ids], probabilities[:, ids], thresholds)
            if (b + 1) % 250 == 0:
                log(f"测试集分层配对Bootstrap：{b+1}/{repeats}")
        np.savez_compressed(path, 指标分布=distributions)
    confidence, effects = [], []
    for j, key in enumerate(keys):
        for m, name in enumerate(names):
            low, high = np.nanquantile(distributions[:, j, m], [.025, .975])
            confidence.append({"键": key, "模型": 模型名称[fitted[key].code], "策略": fitted[key].strategy,
                               "指标": 指标名称[name], "估计值": float(table.set_index("键").loc[key, name]),
                               "95%下限": low, "95%上限": high,
                               "有效重复数": int(np.isfinite(distributions[:, j, m]).sum())})
    for code in 模型名称:
        a, b = keys.index(f"{code}|SMOTE-ENC"), keys.index(f"{code}|未采样")
        for m, name in enumerate(names):
            delta = distributions[:, a, m] - distributions[:, b, m]
            low, high = np.nanquantile(delta, [.025, .975])
            t = table.set_index("键")
            effects.append({"模型": 模型名称[code], "指标": 指标名称[name],
                            "差值_SMOTEENC减未采样": t.loc[keys[a], name] - t.loc[keys[b], name],
                            "差值95%下限": low, "差值95%上限": high, "有效重复数": int(np.isfinite(delta).sum())})
    ci, effect = pd.DataFrame(confidence), pd.DataFrame(effects)
    csv(ci, out / "表格" / "主分析_指标置信区间.csv")
    csv(pd.DataFrame(effects), out / "表格" / "SMOTEENC相对未采样的配对效应.csv")
    return ci, effect


def shap_analysis(fit, d, train, dev, test, out, stability=100, workers=4):
    rng = np.random.default_rng(种子)
    background_ids = rng.choice(train, min(128 if fit.code == "LR" else 32, len(train)), replace=False)
    bg = d.iloc[background_ids].copy()
    names = fit.preprocessor.raw_names_
    # 测试集对最终冻结模型的解释，不用于调参或变量选择。
    values, base, info = explain_calibrated(fit, d.iloc[test], bg)
    csv(pd.DataFrame(values, columns=names).assign(记录编号=d.loc[test, "记录编号"].to_numpy()),
        out / "私密记录_禁止公开" / "测试集全部原始变量SHAP.csv")
    csv(pd.DataFrame({"记录编号": d.loc[test, "记录编号"].to_numpy(), "基线": base,
                      "完整映射": fit.decision_function(d.iloc[test]),
                      "校准后概率": fit.predict_proba(d.iloc[test])}),
        out / "私密记录_禁止公开" / "SHAP基线与完整映射.csv")
    csv(d.loc[background_ids, ["记录编号"]], out / "私密记录_禁止公开" / "SHAP背景索引.csv")
    importance = pd.DataFrame({"变量": names, "平均绝对SHAP": np.mean(np.abs(values), axis=0)})
    importance["名次"] = importance["平均绝对SHAP"].rank(method="min", ascending=False).astype(int)
    importance = importance.sort_values("平均绝对SHAP", ascending=False)
    csv(importance, out / "表格" / "主模型全部变量SHAP重要性.csv")
    probability = fit.predict_proba(d.iloc[test])
    choice = []
    for flag in [True, False]:
        candidates = np.flatnonzero((probability >= fit.threshold) == flag)
        if not len(candidates):
            raise RuntimeError("缺少预先要求的预测阳性或阴性案例，不能捏造瀑布图。")
        i = int(rng.choice(candidates))
        choice.append(i)
    cases = [{"案例": "预测阳性案例" if j == 0 else "预测阴性案例", "系统标签": int(d.iloc[test[i]]["标签"]),
              "模型概率": float(probability[i]), "固定阈值": float(fit.threshold),
              "基础log_odds": float(base[i]), "最终log_odds": float(fit.decision_function(d.iloc[test[[i]]])[0]),
              "解释贡献和": float(values[i].sum())} for j, i in enumerate(choice)]
    csv(pd.DataFrame(cases), out / "表格" / "瀑布图案例说明.csv")
    write_json(out / "复现记录" / "SHAP解释定义与可加性核查.json", dict(info,
        背景来源="真实训练记录；不使用合成样本", 主模型=f"{fit.code}|{fit.strategy}",
        分类变量口径="独热列SHAP求和后按原始分类变量排名，不把各类别当独立心理维度",
        案例抽取="种子42，按预测阳性和预测阴性随机各抽一例，不按预测是否正确筛选",
        因果解释="SHAP描述模型输出的归因关联，不识别干预因果效应"))
    # 固定开发样本与真实训练背景；每次重采样重新拟合全部步骤。
    target_ids = rng.choice(dev, min(128 if fit.code == "LR" else 32, len(dev)), replace=False)
    fixed = d.iloc[target_ids].copy()
    reference_values, _, _ = explain_calibrated(fit, fixed, bg)
    if fit.code != "LR":
        higher, _, _ = explain_calibrated(fit, fixed.iloc[:16], bg, permutations=16)
        first = np.mean(np.abs(reference_values[:16]), axis=0)
        second = np.mean(np.abs(higher), axis=0)
        write_json(out / "复现记录" / "SHAP近似预算诊断.json", {
            "预算": [8, 16], "固定开发记录数": min(16, len(fixed)),
            "平均绝对贡献排序Spearman": float(stats.spearmanr(first, second).statistic),
            "平均绝对贡献最大差": float(np.max(np.abs(first-second))),
            "口径": "开发样本固定预算诊断；不据测试解释调整模型，近似不确定性如实报告"})
    csv(d.loc[target_ids, ["记录编号"]], out / "私密记录_禁止公开" / "稳定性固定解释样本索引.csv")
    checkpoint = out / "私密记录_禁止公开" / "模型与断点" / "SHAP稳定性"
    checkpoint.mkdir(exist_ok=True)
    def one(b):
        path = checkpoint / f"第{b+1:03d}次.npz"
        if path.exists():
            with np.load(path) as n:
                return n["平均绝对贡献"], n["名次"], n["方向一致率"]
        r = np.random.default_rng(种子 + 10000 + b)
        tr_idx = train[bootstrap_indices(d.loc[train, "标签"].to_numpy(), r)]
        de_idx = dev[bootstrap_indices(d.loc[dev, "标签"].to_numpy(), r)]
        boot = FittedExperiment(fit.code, fit.strategy, fit.params, fit.ratio, fit.k, fit.numeric,
                                种子 + 10000 + b).fit(d.iloc[tr_idx], d.loc[tr_idx, "标签"].to_numpy(),
                                                     d.iloc[de_idx], d.loc[de_idx, "标签"].to_numpy())
        phi, _, _ = explain_calibrated(boot, fixed, bg, seed=种子, permutations=8)
        absolute = np.mean(np.abs(phi), axis=0)
        rank = np.empty(len(absolute), dtype=int)
        rank[np.argsort(-absolute, kind="stable")] = np.arange(1, len(absolute)+1)
        agreement = np.mean(np.sign(phi) == np.sign(reference_values), axis=0)
        np.savez_compressed(path, 平均绝对贡献=absolute, 名次=rank, 方向一致率=agreement)
        return absolute, rank, agreement
    boot_results = []
    # 模型无关SHAP需要较多内存；非线性模型最多两个并发任务。
    with ThreadPoolExecutor(max_workers=workers if fit.code == "LR" else min(2, workers)) as ex:
        for b, result in enumerate(ex.map(one, range(stability)), 1):
            boot_results.append(result)
            if b % 10 == 0:
                log(f"SHAP全流程重拟合稳定性：{b}/{stability}")
    absolute, ranks, agreements = np.asarray(boot_results).transpose(1, 0, 2)
    stability_table = pd.DataFrame({"变量": names, "前5入选频率": (ranks <= 5).mean(axis=0),
                                    "前10入选频率": (ranks <= 10).mean(axis=0), "平均名次": ranks.mean(axis=0),
                                    "名次中位数": np.median(ranks, axis=0),
                                    "名次2.5%分位数": np.quantile(ranks, .025, axis=0),
                                    "名次97.5%分位数": np.quantile(ranks, .975, axis=0),
                                    "同一记录贡献方向一致率": agreements.mean(axis=0),
                                    "平均绝对SHAP中位数": np.median(absolute, axis=0)})
    stability_table = stability_table.merge(importance[["变量", "名次"]], on="变量").sort_values("名次")
    csv(stability_table, out / "表格" / "SHAP全流程重拟合稳定性.csv")
    write_json(out / "复现记录" / "SHAP稳定性设置.json", {"重拟合次数": stability, "固定解释人数": len(fixed),
        "固定背景人数": len(bg), "训练与开发重采样": "在各自角色内独立分层有放回抽样；模型参数固定；重新插补、标准化、编码、采样、拟合和校准",
        "方向定义": "同一固定开发记录中SHAP贡献符号与原模型符号的一致比例；零贡献与零贡献视为一致",
        "不确定性边界": "包含重拟合变异，不包含重新调参选择的不确定性；模型无关SHAP为固定预算近似"})
    csv(pd.DataFrame(ranks, columns=names).assign(重拟合次数=np.arange(1, stability+1)),
        out / "表格" / "SHAP重拟合逐次名次.csv")
    return values, base, importance, choice, stability_table, info


def sensitivity_analysis(fitted, selected_key, d, train, dev, test, folds, out, workers, repeats=2000):
    code = selected_key.split("|")[0]
    numeric = [c for c in 连续变量 if c != "自伤行为"]
    sensitivity, _ = fit_comparison(d, train, dev, folds, numeric, [code], out, workers, "去除自伤")
    # 两种策略的重新调参、校准、阈值全部在测试集求值前结束。
    write_json(out / "复现记录" / "去除自伤方案冻结.json", {
        "模型家族": 模型名称[code], "连续预测变量数": len(numeric), "分类预测变量数": len(分类变量),
        "标签和数据划分": "与主分析完全一致", "参数与阈值": {k: {"参数": f.params, "阈值": f.threshold,
            "采样比例": f.ratio, "近邻数": f.k} for k, f in sensitivity.items()}})
    table, probs, _ = test_evaluation(sensitivity, d, test, out, "去除自伤")
    y = d.loc[test, "标签"].to_numpy()
    comparison, ranks = [], []
    rng = np.random.default_rng(种子)
    bg = d.iloc[rng.choice(train, min(128, len(train)), replace=False)]
    for strategy in 策略名称:
        key = f"{code}|{strategy}"
        main, reduced = fitted[key], sensitivity[key]
        p0, p1 = main.predict_proba(d.iloc[test]), probs[key]
        m0, m1 = metrics(y, p0, main.threshold), metrics(y, p1, reduced.threshold)
        rng_ci = np.random.default_rng(20261022)
        delta_dist = np.empty((repeats, len(指标名称)))
        for b in range(repeats):
            ids = bootstrap_indices(y, rng_ci)
            pair = fast_metric_array(y[ids], np.array([p0[ids], p1[ids]]), [main.threshold, reduced.threshold])
            delta_dist[b] = pair[1] - pair[0]
        np.savez_compressed(out / "私密记录_禁止公开" / "模型与断点" / f"去除自伤_{strategy}_配对分布.npz", 差值分布=delta_dist)
        for name in 指标名称:
            vals = delta_dist[:, list(指标名称).index(name)]
            low, high = np.nanquantile(vals, [.025, .975])
            comparison.append({"模型": 模型名称[code], "策略": strategy, "指标": 指标名称[name],
                               "保留自伤": m0[name], "去除自伤": m1[name], "去除减保留": m1[name] - m0[name],
                               "差值95%下限": low, "差值95%上限": high, "有效重复数": int(np.isfinite(vals).sum())})
        for label, fit in [("保留自伤", main), ("去除自伤", reduced)]:
            val, _, _ = explain_calibrated(fit, d.iloc[test], bg)
            mean = np.mean(np.abs(val), axis=0)
            for c, v, rank in zip(fit.preprocessor.raw_names_, mean, stats.rankdata(-mean, method="min")):
                ranks.append({"策略": strategy, "分析": label, "变量": c, "平均绝对SHAP": v, "名次": rank})
    csv(pd.DataFrame(comparison), out / "表格" / "去除自伤前后测试表现比较.csv")
    csv(pd.DataFrame(ranks), out / "表格" / "去除自伤前后解释排序比较.csv")
    return table


def main():
    parser = argparse.ArgumentParser(description="大学生自杀意念可解释机器学习完整实验")
    parser.add_argument("--数据目录", default=str(Path(__file__).resolve().parents[2] / "data" / "private" / "original"))
    parser.add_argument("--结果目录", default=str(Path(__file__).resolve().parents[2] / "results" / "generated" / "zh"))
    parser.add_argument("--并行数", type=int, default=4)
    parser.add_argument("--重采样次数", type=int, default=2000)
    parser.add_argument("--稳定性次数", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.结果目录).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ["表格", "图片", "复现记录", "私密记录_禁止公开", "私密记录_禁止公开/模型与断点"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = [{"文件": f"{year}.xlsx", "字节数": (Path(args.数据目录)/f"{year}.xlsx").stat().st_size,
                "SHA256": digest(Path(args.数据目录)/f"{year}.xlsx")} for year in range(2021, 2026)]
    versions = {pkg: importlib.metadata.version(pkg) for pkg in
                ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "shap", "matplotlib", "openpyxl", "joblib"]}
    code_hashes = {p.name: digest(p) for p in sorted(代码目录.iterdir()) if p.suffix in [".py", ".mjs"]}
    sig_content = {"输入": sources, "算法SHA256": digest(代码目录 / "算法与统计.py"), "执行源码SHA256": code_hashes,
                   "种子": 种子, "五折": 5, "划分比例": [.6, .2, .2], "软件版本": versions,
                   "搜索": {c: parameter_stages(c) for c in 模型名称},
                   "重复次数": [args.重采样次数, args.稳定性次数]}
    signature = hashlib.sha256(json.dumps(sig_content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "复现记录" / "研究配置.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["研究签名"] != signature:
            raise RuntimeError("输入、算法或实验配置已经变化；请使用新的独立结果目录，避免混用旧断点。")
    else:
        write_json(manifest_path, dict(sig_content, 研究签名=signature, 数据目录=str(Path(args.数据目录).resolve()),
                                      结果目录=str(out), 代码目录=str(代码目录)))
    # 断点指纹通过后才读取并写出本轮清洗、划分和分析产物。
    from 字段核对 import audit_inputs
    audit_inputs(args.数据目录, out)
    d, audit, sources = read_data(args.数据目录, out)
    train, dev, test, folds = make_split(d, out)
    log(f"五年合并读取完成：{len(d)}条；标签阳性{int(d['标签'].sum())}条。划分{len(train)}/{len(dev)}/{len(test)}。")
    write_json(out / "复现记录" / "运行环境版本.json", dict(versions, Python=platform.python_version(), 操作系统=platform.platform()))
    with threadpool_limits(limits=1):
        fitted, development = fit_comparison(d, train, dev, folds, 连续变量, list(模型名称), out, args.并行数)
        selection = freeze_selection(development, out, signature)
        spearman = describe(d, train, out)
        table, probs, bins = test_evaluation(fitted, d, test, out)
        ci, effect = paired_bootstrap(fitted, probs, d.loc[test, "标签"].to_numpy(), table, out, args.重采样次数)
        selected = selection["主模型键"]
        same_strategy = [f"{c}|{selection['主策略']}" for c in 模型名称]
        delong = delong_compare(d.loc[test, "标签"].to_numpy(), [probs[k] for k in same_strategy],
                                [模型名称[fitted[k].code] for k in same_strategy], selection["主模型"])
        csv(delong, out / "表格" / "主模型DeLong检验与Holm校正.csv")
        shap_results = shap_analysis(fitted[selected], d, train, dev, test, out, args.稳定性次数, args.并行数)
        sensitivity = sensitivity_analysis(fitted, selected, d, train, dev, test, folds, out, args.并行数, args.重采样次数)
        baseline = metrics(d.loc[test, "标签"].to_numpy(), np.full(len(test), d.loc[train, "标签"].mean()), .5)
        csv(pd.DataFrame([baseline]).rename(columns=指标名称), out / "表格" / "训练阳性率常数概率参照.csv")
        from 图表输出 import create_all_figures, write_report
        plot_args = (out, d, audit, train, dev, test, spearman, fitted, probs, table, ci, bins, selected, shap_results)
        joblib.dump(plot_args, out / "私密记录_禁止公开" / "绘图固定输入.joblib", compress=3)
        create_all_figures(*plot_args)
        report = write_report(out, table, ci, effect, selection, fitted, shap_results[-1], sensitivity,
                              len(d), len(test), time.monotonic() - started)
        log(report)
    log("全部实验与输出检查完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
