"""独立核验冻结输出，不导入建模及原指标计算代码，不重新拟合模型。

AUC使用平均秩U统计量，AP按完整并列概率组累积；DeLong使用独立
midrank实现。正式复核会独立重建全部测试Bootstrap分布。原始Excel
只作字节校验，不解析或输出个体资料。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

CODE = Path(__file__).resolve().parent
if (CODE / "依赖环境").exists():
    sys.path.insert(0, str(CODE / "依赖环境"))

import numpy as np
import pandas as pd
from PIL import Image

MODELS = {"LR": "逻辑回归", "DT": "决策树", "RF": "随机森林",
          "ET": "极端随机树", "XGB": "XGBoost", "LGBM": "LightGBM"}
STRATEGIES = ["未采样", "SMOTE-ENC"]
METRICS = {"AUC": "ROC-AUC", "AP": "平均精确率AP", "Recall": "召回率",
           "Specificity": "特异度", "PPV": "阳性预测值", "NPV": "阴性预测值",
           "F1": "F1值", "Accuracy": "准确率", "BalancedAccuracy": "平衡准确率",
           "Brier": "Brier分数", "FlagRate": "筛查阳性比例", "Review1000": "每千条记录复核数",
           "Miss1000": "每千条记录漏识标签阳性数", "ReviewPerPositive": "每识别一例标签阳性的复核数"}
RAW_FEATURES = ["幻觉妄想", "焦虑", "抑郁", "偏执", "自卑", "敏感", "社交恐惧", "躯体化",
                "依赖", "敌对攻击", "冲动", "强迫", "网络成瘾", "自伤行为", "进食问题", "睡眠困扰",
                "学校适应困难", "人际关系困扰", "学业压力", "就业问题", "恋爱困扰",
                "性别", "民族", "生源地", "是否独生"]
EXPECTED_KEYS = [f"{code}|{strategy}" for code in MODELS for strategy in STRATEGIES]
FIGURES = ["图01_数据纳入与实验流程", "图02_心理维度Spearman相关热图", "图03_六模型召回率对比",
           "图04_六模型ROC曲线", "图05_六模型PR曲线", "图06_六模型特异度对比", "图07_主模型概率校准曲线",
           "图08_主模型混淆矩阵", "图09_主模型SHAP蜂群图_前十变量", "图10_主模型SHAP瀑布图_前五变量",
           "图11_SHAP重要性重拟合稳定性"]
REPEATS, STABILITY, BOOT_SEED = 2000, 100, 20261022


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    # 概率及阈值须恢复为保存时的浮点数，避免读入造成阈值边界并列。
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip",
                       dtype={"记录编号": "string"})


def require_columns(frame, columns):
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError("缺少字段：" + "、".join(sorted(absent)))


class Audit:
    def __init__(self, out):
        self.out = Path(out)
        self.checks = []

    def check(self, name, passed, detail=""):
        passed = bool(passed)
        self.checks.append({"检查项": name, "通过": passed, "说明": str(detail)})
        if not passed:
            print(f"失败：{name}；{detail}", flush=True)
        return passed

    def close(self, name, actual, expected, atol=1e-10, rtol=5e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape:
            return self.check(name, False, f"形状不同：{a.shape} / {b.shape}")
        passed = (not np.isinf(a).any() and not np.isinf(b).any()
                  and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
        both = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        return self.check(name, passed,
                          f"最大绝对差={maximum:.6g}；缺失位置一致={np.array_equal(np.isnan(a), np.isnan(b))}")

    def section(self, name, function):
        try:
            return function()
        except Exception as exc:
            self.check(name, False, f"{type(exc).__name__}: {exc}")
            return None

    def numeric(self, name, frame):
        values = frame.select_dtypes(include="number").to_numpy(dtype=float)
        self.check(name, not np.isinf(values).any(), "允许不可估计NaN；不允许正负无穷数")

    def finish(self):
        frame = pd.DataFrame(self.checks, columns=["检查项", "通过", "说明"])
        frame.to_csv(self.out / "独立复核清单.csv", index=False, encoding="utf-8-sig")
        failed = [c for c in self.checks if not c["通过"]]
        conclusion = {"通过": not failed, "检查数": len(self.checks), "通过数": len(self.checks) - len(failed),
                      "失败数": len(failed), "失败项": failed,
                      "复核时间UTC": datetime.now(timezone.utc).isoformat(),
                      "复核脚本SHA256": sha256(__file__),
                      "方法": {"AUC": "并列值平均秩U统计量", "AP": "并列概率组逐阶累积",
                               "DeLong": "独立midrank实现及五项Holm校正",
                               "Bootstrap": "按固定顺序和种子独立重建全部2000次指标与配对差值",
                               "不可估计指标": "零分母为NaN，要求汇总及逐次分布缺失位置一致"}}
        path = self.out / "复现记录" / "独立复核结论.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(conclusion, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"独立复核：{len(self.checks) - len(failed)}/{len(self.checks)}项通过。", flush=True)
        if failed:
            raise AssertionError("输出独立复核失败，见独立复核清单.csv及独立复核结论.json")


def midranks(values):
    """并列平均秩，秩从1起；不依赖建模代码或scipy.rankdata。"""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    ranked = np.repeat((starts + ends + 1.0) / 2.0, ends - starts)
    result = np.empty(len(x), dtype=float)
    result[order] = ranked
    return result


def independent_metrics(y, probability, threshold):
    """独立计算全部14项指标和四格人数，零分母返回NaN。"""
    original_y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if len(original_y) != len(p) or not len(original_y) or not np.isin(original_y, [0, 1]).all():
        raise ValueError("标签或预测长度无效")
    y = original_y.astype(int)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isfinite(threshold):
        raise ValueError("概率或阈值无效")
    positive, predicted = y == 1, p >= threshold
    tp, fn = int((positive & predicted).sum()), int((positive & ~predicted).sum())
    fp, tn = int((~positive & predicted).sum()), int((~positive & ~predicted).sum())
    n1, n0, n = tp + fn, tn + fp, len(y)
    div = lambda numerator, denominator: float(numerator / denominator) if denominator else np.nan
    ranks = midranks(p)
    auc = div(float(ranks[positive].sum()) - n1 * (n1 + 1) / 2, n1 * n0)
    order = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1, n]
    cumulative_tp = np.cumsum(sorted_y)[ends - 1]
    increments = np.diff(np.r_[0, cumulative_tp])
    ap = div(float(np.sum((cumulative_tp / ends) * increments)), n1)
    recall, specificity = div(tp, n1), div(tn, n0)
    return {"AUC": auc, "AP": ap, "Recall": recall, "Specificity": specificity,
            "PPV": div(tp, tp + fp), "NPV": div(tn, tn + fn),
            "F1": div(2 * tp, 2 * tp + fp + fn), "Accuracy": div(tp + tn, n),
            "BalancedAccuracy": (recall + specificity) / 2,
            "Brier": float(np.dot(p - y, p - y) / n), "FlagRate": div(tp + fp, n),
            "Review1000": div(1000 * (tp + fp), n), "Miss1000": div(1000 * fn, n),
            "ReviewPerPositive": div(tp + fp, tp),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Threshold": float(threshold)}


def audit_config(audit):
    out = audit.out
    config = json.loads((out / "复现记录" / "研究配置.json").read_text(encoding="utf-8"))
    audit.check("正式重复次数为2000和100", config["重复次数"] == [REPEATS, STABILITY], str(config["重复次数"]))
    audit.check("固定种子五折及划分比例", config.get("种子") == 42 and config.get("五折") == 5
                and config.get("划分比例") == [.6, .2, .2], "种子42；训练内五折；60/20/20")
    metadata = {"研究签名", "数据目录", "结果目录", "代码目录"}
    payload = {k: v for k, v in config.items() if k not in metadata}
    signature = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    audit.check("运行配置研究签名可重算", signature == config["研究签名"], "规范JSON内容SHA256")
    code_dir = Path(config["代码目录"]).resolve()
    audit.check("复核脚本属于本次代码目录", CODE.resolve() == code_dir, str(code_dir))
    sources = config["执行源码SHA256"]
    actual_sources = {p.name: p for p in code_dir.iterdir() if p.is_file() and p.suffix.lower() in {".py", ".mjs"}}
    audit.check("源码指纹覆盖全部执行源码", set(sources) == set(actual_sources),
                f"登记{len(sources)}个；目录{len(actual_sources)}个.py/.mjs文件")
    post_file = out / "复现记录" / "后处理源码版本.json"
    post = json.loads(post_file.read_text(encoding="utf-8")) if post_file.exists() else {}
    revised = post.get("允许变更源码", {})
    if post:
        audit.check("后处理绑定原冻结研究", post.get("研究签名") == config["研究签名"], "原研究配置及模型冻结签名不改写")
        audit.check("后处理仅涉及图形与复核源码", set(revised).issubset({"图表输出.py", "工作簿导出.mjs", "输出复核.py"}),
                    "禁止借后处理修改读取、建模、校准、阈值、模型选择或统计计算源码")
    for name, expected in sources.items():
        def check_code(name=name, expected=expected):
            current = sha256(actual_sources[name])
            if current == expected:
                audit.check(f"源码SHA256/{name}", True, "与初始执行指纹完全一致")
            elif name in revised and name in {"图表输出.py", "工作簿导出.mjs", "输出复核.py"}:
                archive = out / "复现记录" / "初始执行源码" / name
                passed = sha256(archive) == expected and current == revised[name]
                audit.check(f"源码SHA256/{name}", passed, "初始执行源码存档与原指纹一致；当前后处理版本另行绑定，变更已披露")
            else:
                audit.check(f"源码SHA256/{name}", False, "分析源码变化且不属于有记录的图形后处理")
        audit.section(f"源码指纹读取/{name}", check_code)
    for rel, expected in post.get("冻结数值结果SHA256", {}).items():
        def check_numeric_file(rel=rel, expected=expected):
            path = (out / rel).resolve()
            if not path.is_relative_to(out.resolve()):
                raise ValueError("数值校验路径超出结果目录")
            audit.check(f"图形续跑前后数值字节一致/{rel}", sha256(path) == expected,
                        "建模、逐例预测、解释、配对分布及统计表保持原冻结字节")
        audit.section(f"冻结数值校验/{rel}", check_numeric_file)
    if "算法SHA256" in config:
        audit.check("算法指纹与源码清单一致", config["算法SHA256"] == sources.get("算法与统计.py"), "双重登记一致")
    original = config["输入"]
    audit.check("原始输入固定为五个年份", len(original) == 5 and {s["文件"] for s in original} ==
                {f"{year}.xlsx" for year in range(2021, 2026)}, "2021—2025.xlsx")
    recorded = load_csv(out / "复现记录" / "原始文件校验值.csv")
    require_columns(recorded, ["文件", "字节数", "SHA256"])
    audit.check("原始文件校验表唯一且数量一致", recorded["文件"].is_unique and len(recorded) == len(original), "与配置对应")
    indexed = recorded.set_index("文件")
    input_dir = Path(config["数据目录"]).resolve()
    for item in original:
        def check_source(item=item):
            name = item["文件"]
            if Path(name).name != name:
                raise ValueError("输入文件名包含目录")
            path = input_dir / name
            match = (path.stat().st_size == int(item["字节数"]) == int(indexed.loc[name, "字节数"])
                     and sha256(path) == item["SHA256"] == indexed.loc[name, "SHA256"])
            audit.check(f"原始文件字节数及SHA256/{name}", match, "只读字节校验，不解析个体资料")
        audit.section(f"原始文件核验/{item['文件']}", check_source)
    frozen = json.loads((out / "复现记录" / "测试集启封前的冻结方案.json").read_text(encoding="utf-8"))
    audit.check("冻结方案绑定当前研究签名", frozen["研究签名"] == config["研究签名"], "禁止混用其他运行")
    key = frozen["主模型键"]
    audit.check("冻结模型和策略对应", key in EXPECTED_KEYS and
                frozen["主模型"] == MODELS[key.split("|")[0]] and frozen["主策略"] == key.split("|")[1],
                "模型键、模型名和策略一致")
    return config, frozen


def audit_predictions(audit, prefix, keys, target, reference_ids=None):
    summary = load_csv(audit.out / "表格" / f"{prefix}_测试集完整指标.csv")
    records = load_csv(audit.out / "私密记录_禁止公开" / f"{prefix}_测试集逐记录预测.csv")
    require_columns(summary, ["键", "模型", "策略"] + list(METRICS.values()) + ["TN", "FP", "FN", "TP", "Threshold"])
    require_columns(records, ["记录编号", "系统标签", "模型", "策略", "校准前概率", "校准后概率", "固定阈值", "预测标签"])
    audit.numeric(f"{prefix}汇总及逐记录无无穷数", pd.concat([summary, records], ignore_index=True))
    audit.check(f"{prefix}汇总模型顺序与分布约定一致", summary["键"].tolist() == keys, f"预期{len(keys)}组")
    observed = set(zip(records["模型"], records["策略"]))
    expected = {(MODELS[k.split("|")[0]], k.split("|")[1]) for k in keys}
    audit.check(f"{prefix}逐记录模型分组完整", observed == expected, f"预期{len(keys)}组")
    audit.check(f"{prefix}汇总键唯一", summary["键"].is_unique, "索引转换前明确核对唯一性")
    summary = summary.set_index("键")
    output = {"table": summary, "groups": {}, "metrics": {}, "keys": keys}
    for key in keys:
        def one(key=key):
            code, strategy = key.split("|")
            model = MODELS[code]
            group = records[(records["模型"] == model) & (records["策略"] == strategy)].copy()
            name = f"{prefix}/{model}/{strategy}"
            ids = group["记录编号"].tolist()
            audit.check(f"{name}测试索引唯一且完整", group["记录编号"].is_unique and
                        set(ids) == set(target.index) and len(ids) == len(target), "与固定测试划分核对")
            if reference_ids is not None:
                audit.check(f"{name}测试顺序固定", ids == reference_ids, "Bootstrap按相同记录顺序执行")
            elif "ids" in output:
                audit.check(f"{name}测试顺序相同", ids == output["ids"], "十二组预测逐项对齐")
            if "ids" not in output:
                output["ids"] = ids
            original_y = group["系统标签"].to_numpy()
            audit.check(f"{name}系统标签仅含0和1", np.isin(original_y, [0, 1]).all(), "禁止整数转换掩盖非法标签")
            y = original_y.astype(int)
            audit.check(f"{name}系统标签一致", np.isin(original_y, [0, 1]).all() and
                        np.array_equal(target.reindex(ids).to_numpy(), y), "只按索引比对，不输出个体信息")
            thresholds = group["固定阈值"].to_numpy(dtype=float)
            if not len(thresholds):
                raise ValueError("预测分组为空")
            audit.check(f"{name}阈值恒定且合法", np.isfinite(thresholds).all() and
                        np.all(thresholds == thresholds[0]) and 0 <= thresholds[0] <= 1, "每组一个冻结阈值")
            for phase in ["校准前概率", "校准后概率"]:
                p = group[phase].to_numpy(dtype=float)
                audit.check(f"{name}/{phase}范围", np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "有限且位于[0,1]")
            p = group["校准后概率"].to_numpy(dtype=float)
            recalculated = independent_metrics(y, p, thresholds[0])
            audit.check(f"{name}预测标签与固定阈值一致",
                        np.array_equal(group["预测标签"].to_numpy(), (p >= thresholds[0]).astype(int)), "整体保留阈值并列")
            saved = summary.loc[key]
            audit.check(f"{name}汇总模型名称策略对应", saved["模型"] == model and saved["策略"] == strategy, "按模型键核验")
            columns = [METRICS.get(k, k) for k in recalculated]
            tolerance = np.full(len(recalculated), 1e-10)
            if code == "XGB":
                # 原XGBoost校准概率及sklearn Brier归约为float32；其余点指标和所有区间仍严格核对。
                p32 = p.astype(np.float32)
                brier32 = float(np.mean((p32 - y.astype(np.float32)) ** 2))
                witness = np.array_equal(p, p32.astype(np.float64)) and float(saved["Brier分数"]) == brier32
                audit.check(f"{name}Brier单精度来源可精确复现", witness,
                            "逐例概率均可精确恢复float32，原Brier与独立float32均方误差完全相同；float64差异为归约精度")
                tolerance[list(recalculated).index("Brier")] = 2e-8
            audit.close(f"{name}全部14指标及混淆矩阵独立复算", saved[columns].to_numpy(dtype=float),
                        list(recalculated.values()), atol=tolerance)
            output["groups"][key] = group
            output["metrics"][key] = recalculated
        audit.section(f"{prefix}/{key}预测复算异常", one)
    return output


def percentile(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).any():
        return np.array([np.nan, np.nan])
    return np.nanquantile(x, [.025, .975])


def draw_bootstrap(y, rng):
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate((rng.choice(positive, size=len(positive), replace=True),
                           rng.choice(negative, size=len(negative), replace=True)))


def audit_bootstrap(audit, primary, ci, effect):
    path = audit.out / "私密记录_禁止公开" / "模型与断点" / "主分析_配对重采样分布.npz"
    with np.load(path, allow_pickle=False) as saved:
        distributions = saved["指标分布"].copy()
    if not audit.check("主分析Bootstrap分布形状", distributions.shape == (REPEATS, 12, len(METRICS)), str(distributions.shape)):
        return None
    audit.check("主分析Bootstrap分布无无穷数", not np.isinf(distributions).any(), "NaN仅用于零分母指标")
    require_columns(ci, ["键", "模型", "策略", "指标", "估计值", "95%下限", "95%上限", "有效重复数"])
    require_columns(effect, ["模型", "指标", "差值_SMOTEENC减未采样", "差值95%下限", "差值95%上限", "有效重复数"])
    audit.numeric("主分析区间及配对效应无无穷数", pd.concat([ci, effect], ignore_index=True))
    audit.check("主分析指标区间条目完整且唯一", len(ci) == 12 * len(METRICS) and
                not ci.duplicated(["键", "指标"]).any() and
                set(zip(ci["键"], ci["指标"])) == {(k, m) for k in EXPECTED_KEYS for m in METRICS.values()}, "每组14项区间")
    audit.check("同模型采样配对效应完整且唯一", len(effect) == 6 * len(METRICS) and
                not effect.duplicated(["模型", "指标"]).any() and
                set(zip(effect["模型"], effect["指标"])) == {(n, m) for n in MODELS.values() for m in METRICS.values()}, "六模型各14项")
    ci_index, effect_index = ci.set_index(["键", "指标"]), effect.set_index(["模型", "指标"])
    for j, key in enumerate(EXPECTED_KEYS):
        def confidence(j=j, key=key):
            code, strategy = key.split("|")
            saved_rows, expected_rows = [], []
            for m, (metric, title) in enumerate(METRICS.items()):
                row = ci_index.loc[(key, title)]
                low, high = percentile(distributions[:, j, m])
                saved_rows.append([row["估计值"], row["95%下限"], row["95%上限"], row["有效重复数"]])
                expected_rows.append([primary["metrics"][key][metric], low, high, np.isfinite(distributions[:, j, m]).sum()])
                if row["模型"] != MODELS[code] or row["策略"] != strategy:
                    raise ValueError("区间表模型名或策略与键不一致")
            tolerance = np.full((len(METRICS), 4), 1e-10)
            if code == "XGB":
                tolerance[list(METRICS).index("Brier"), 0] = 2e-8
            audit.close(f"{key}全部区间分位数及有效次数", saved_rows, expected_rows, atol=tolerance)
        audit.section(f"{key}区间复算异常", confidence)
    for code, model in MODELS.items():
        def paired(code=code, model=model):
            a, b = EXPECTED_KEYS.index(f"{code}|SMOTE-ENC"), EXPECTED_KEYS.index(f"{code}|未采样")
            saved_rows, expected_rows = [], []
            for m, (metric, title) in enumerate(METRICS.items()):
                values = distributions[:, a, m] - distributions[:, b, m]
                row = effect_index.loc[(model, title)]
                low, high = percentile(values)
                delta = primary["metrics"][EXPECTED_KEYS[a]][metric] - primary["metrics"][EXPECTED_KEYS[b]][metric]
                saved_rows.append([row["差值_SMOTEENC减未采样"], row["差值95%下限"], row["差值95%上限"], row["有效重复数"]])
                expected_rows.append([delta, low, high, np.isfinite(values).sum()])
            tolerance = np.full((len(METRICS), 4), 1e-10)
            if code == "XGB":
                tolerance[list(METRICS).index("Brier"), 0] = 4e-8
            audit.close(f"{model}全部采样配对差值及区间", saved_rows, expected_rows, atol=tolerance)
        audit.section(f"{model}配对效应复算异常", paired)
    y = primary["groups"][EXPECTED_KEYS[0]]["系统标签"].to_numpy(dtype=int)
    p = np.array([primary["groups"][key]["校准后概率"].to_numpy(dtype=float) for key in EXPECTED_KEYS])
    thresholds = [primary["metrics"][key]["Threshold"] for key in EXPECTED_KEYS]
    rng, computed = np.random.default_rng(BOOT_SEED), np.empty_like(distributions)
    for repeat in range(REPEATS):
        ids = draw_bootstrap(y, rng)
        for j in range(12):
            metrics = independent_metrics(y[ids], p[j, ids], thresholds[j])
            computed[repeat, j] = [metrics[metric] for metric in METRICS]
        if (repeat + 1) % 250 == 0:
            print(f"主分析Bootstrap独立逐次复算：{repeat + 1}/{REPEATS}", flush=True)
    audit.close("主分析全部2000次配对指标分布独立复算", distributions, computed)
    return distributions


def audit_sensitivity(audit, primary, reduced, frozen):
    comparison = load_csv(audit.out / "表格" / "去除自伤前后测试表现比较.csv")
    require_columns(comparison, ["模型", "策略", "指标", "保留自伤", "去除自伤", "去除减保留",
                                 "差值95%下限", "差值95%上限", "有效重复数"])
    model, code = frozen["主模型"], frozen["主模型键"].split("|")[0]
    audit.numeric("去自伤比较无无穷数", comparison)
    audit.check("去自伤两策略全部指标条目完整", len(comparison) == 2 * len(METRICS) and
                comparison["模型"].eq(model).all() and not comparison.duplicated(["策略", "指标"]).any()
                and set(zip(comparison["策略"], comparison["指标"])) ==
                {(s, m) for s in STRATEGIES for m in METRICS.values()}, "固定模型家族，两策略各14项")
    rows = comparison.set_index(["策略", "指标"])
    y = primary["groups"][EXPECTED_KEYS[0]]["系统标签"].to_numpy(dtype=int)
    for strategy in STRATEGIES:
        def one(strategy=strategy):
            key = f"{code}|{strategy}"
            with np.load(audit.out / "私密记录_禁止公开" / "模型与断点" / f"去除自伤_{strategy}_配对分布.npz", allow_pickle=False) as saved:
                distribution = saved["差值分布"].copy()
            if not audit.check(f"去自伤/{strategy}配对分布形状", distribution.shape == (REPEATS, len(METRICS)), str(distribution.shape)):
                return
            audit.check(f"去自伤/{strategy}分布无无穷数", not np.isinf(distribution).any(), "允许不可估计NaN")
            saved_rows, expected_rows = [], []
            for j, (metric, title) in enumerate(METRICS.items()):
                row = rows.loc[(strategy, title)]
                before, after = primary["metrics"][key][metric], reduced["metrics"][key][metric]
                low, high = percentile(distribution[:, j])
                saved_rows.append([row[c] for c in ["保留自伤", "去除自伤", "去除减保留", "差值95%下限", "差值95%上限", "有效重复数"]])
                expected_rows.append([before, after, after - before, low, high, np.isfinite(distribution[:, j]).sum()])
            audit.close(f"去自伤/{strategy}全部点差与95%区间", saved_rows, expected_rows)
            p0 = primary["groups"][key]["校准后概率"].to_numpy(dtype=float)
            p1 = reduced["groups"][key]["校准后概率"].to_numpy(dtype=float)
            t0, t1 = primary["metrics"][key]["Threshold"], reduced["metrics"][key]["Threshold"]
            computed, rng = np.empty_like(distribution), np.random.default_rng(BOOT_SEED)
            for repeat in range(REPEATS):
                ids = draw_bootstrap(y, rng)
                before = independent_metrics(y[ids], p0[ids], t0)
                after = independent_metrics(y[ids], p1[ids], t1)
                computed[repeat] = [after[m] - before[m] for m in METRICS]
                if (repeat + 1) % 500 == 0:
                    print(f"去自伤/{strategy}Bootstrap独立逐次复算：{repeat + 1}/{REPEATS}", flush=True)
            audit.close(f"去自伤/{strategy}全部2000次配对差值独立复算", distribution, computed)
        audit.section(f"去自伤/{strategy}配对复算异常", one)


def independent_delong(y, probabilities):
    """正类优先排列的midrank快速DeLong，返回AUC及完整协方差。"""
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if min(npos, nneg) < 2:
        raise ValueError("DeLong需要每类至少两条记录")
    ordered = np.concatenate((p[:, y == 1], p[:, y == 0]), axis=1)
    tx = np.array([midranks(row[:npos]) for row in ordered])
    ty = np.array([midranks(row[npos:]) for row in ordered])
    tz = np.array([midranks(row) for row in ordered])
    auc = (tz[:, :npos].sum(axis=1) - npos * (npos + 1) / 2) / (npos * nneg)
    vpos = (tz[:, :npos] - tx) / nneg
    vneg = 1 - (tz[:, npos:] - ty) / npos
    covariance = np.cov(vpos, ddof=1) / npos + np.cov(vneg, ddof=1) / nneg
    return auc, covariance


def independent_holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = np.empty(len(pvalues)), 0.0
    for j, i in enumerate(order):
        running = max(running, (len(order) - j) * float(pvalues[i]))
        adjusted[i] = min(1.0, running)
    return adjusted


def audit_delong(audit, primary, frozen):
    saved = load_csv(audit.out / "表格" / "主模型DeLong检验与Holm校正.csv")
    columns = ["AUC差值", "差值标准误", "差值95%下限", "差值95%上限", "原始P值", "Holm校正P值"]
    require_columns(saved, ["参照模型", "对比模型"] + columns)
    audit.numeric("DeLong表无无穷数", saved)
    strategy, reference = frozen["主策略"], frozen["主模型"]
    names = list(MODELS.values())
    audit.check("DeLong固定参照及五项比较完整", len(saved) == 5 and saved["参照模型"].eq(reference).all()
                and saved["对比模型"].is_unique and set(saved["对比模型"]) == set(names) - {reference}, "仅对应相同采样策略")
    y = primary["groups"][EXPECTED_KEYS[0]]["系统标签"].to_numpy(dtype=int)
    p = np.array([primary["groups"][f"{code}|{strategy}"]["校准后概率"].to_numpy(dtype=float) for code in MODELS])
    aucs, covariance = independent_delong(y, p)
    reference_idx = names.index(reference)
    output = []
    for j, model in enumerate(names):
        if j == reference_idx:
            continue
        difference = float(aucs[reference_idx] - aucs[j])
        contrast = np.zeros(6)
        contrast[reference_idx], contrast[j] = 1, -1
        variance = float(contrast @ covariance @ contrast)
        if variance < -1e-12:
            raise ValueError("DeLong差值方差显著为负")
        se = math.sqrt(max(0.0, variance))
        pvalue = math.erfc(abs(difference) / (se * math.sqrt(2))) if se > 1e-15 else (1.0 if abs(difference) < 1e-12 else 0.0)
        output.append({"对比模型": model, "AUC差值": difference, "差值标准误": se,
                       "差值95%下限": difference - 1.96 * se, "差值95%上限": difference + 1.96 * se,
                       "原始P值": pvalue})
    adjusted = independent_holm([row["原始P值"] for row in output])
    for row, value in zip(output, adjusted):
        row["Holm校正P值"] = value
    recalculated = pd.DataFrame(output).set_index("对比模型")
    actual = saved.set_index("对比模型").reindex(recalculated.index)
    audit.close("DeLong五项协方差差值标准误及P值独立复算", actual[columns].to_numpy(), recalculated[columns].to_numpy(), atol=1e-9)
    audit.close("DeLong全部AUC与独立U统计量一致", aucs,
                [primary["metrics"][f"{code}|{strategy}"]["AUC"] for code in MODELS])


def audit_shap(audit, primary, frozen):
    detail = load_csv(audit.out / "私密记录_禁止公开" / "测试集全部原始变量SHAP.csv")
    mapping = load_csv(audit.out / "私密记录_禁止公开" / "SHAP基线与完整映射.csv")
    require_columns(detail, ["记录编号"] + RAW_FEATURES)
    require_columns(mapping, ["记录编号", "基线", "完整映射", "校准后概率"])
    ids = primary["ids"]
    audit.check("SHAP原始变量恰为25项", set(detail.columns) == set(RAW_FEATURES + ["记录编号"]), "不把独热列当独立心理维度")
    for name, frame in [("SHAP明细", detail), ("完整映射", mapping)]:
        audit.check(f"{name}索引唯一完整", frame["记录编号"].is_unique and len(frame) == len(ids)
                    and set(frame["记录编号"]) == set(ids), "与同一固定测试集对应")
    detail = detail.set_index("记录编号").reindex(ids)
    mapping = mapping.set_index("记录编号").reindex(ids)
    phi = detail[RAW_FEATURES].to_numpy(dtype=float)
    arrays = mapping[["基线", "完整映射", "校准后概率"]].to_numpy(dtype=float)
    audit.check("SHAP贡献基线及完整映射均有限", np.isfinite(phi).all() and np.isfinite(arrays).all(), "解释输出不能缺失")
    audit.close("所有测试记录SHAP基线加和还原完整映射", mapping["基线"].to_numpy() + phi.sum(axis=1),
                mapping["完整映射"].to_numpy(), atol=1e-5, rtol=1e-10)
    score = mapping["完整映射"].to_numpy(dtype=float)
    probability = np.empty(len(score), dtype=float)
    positive = score >= 0
    probability[positive] = 1 / (1 + np.exp(-score[positive]))
    exponential = np.exp(score[~positive])
    probability[~positive] = exponential / (1 + exponential)
    selected = primary["groups"][frozen["主模型键"]].set_index("记录编号").reindex(ids)
    audit.close("完整校准映射转换概率一致", mapping["校准后概率"].to_numpy(), probability)
    audit.close("SHAP对应冻结主模型的实际测试概率", mapping["校准后概率"].to_numpy(), selected["校准后概率"].to_numpy())
    importance = load_csv(audit.out / "表格" / "主模型全部变量SHAP重要性.csv")
    require_columns(importance, ["变量", "平均绝对SHAP", "名次"])
    audit.check("SHAP全局重要性变量唯一完整", len(importance) == 25 and importance["变量"].is_unique
                and set(importance["变量"]) == set(RAW_FEATURES), "25个原始变量")
    importance = importance.set_index("变量").reindex(RAW_FEATURES)
    mean = np.mean(np.abs(phi), axis=0)
    audit.close("全局平均绝对SHAP由私密明细独立复算", importance["平均绝对SHAP"].to_numpy(), mean)
    expected_ranks = np.array([1 + int(np.sum(mean > value)) for value in mean])
    audit.close("全局重要性并列名次一致", importance["名次"].to_numpy(), expected_ranks, atol=0, rtol=0)


def audit_stability(audit):
    table = load_csv(audit.out / "表格" / "SHAP全流程重拟合稳定性.csv")
    detail = load_csv(audit.out / "表格" / "SHAP重拟合逐次名次.csv")
    require_columns(detail, ["重拟合次数"] + RAW_FEATURES)
    require_columns(table, ["变量", "前5入选频率", "前10入选频率", "平均名次", "名次中位数", "名次2.5%分位数", "名次97.5%分位数"])
    audit.check("SHAP逐次名次为100次乘25变量", detail.shape == (STABILITY, 26)
                and detail["重拟合次数"].is_unique and set(detail["重拟合次数"]) == set(range(1, STABILITY + 1)), "逐次序号1—100")
    audit.check("SHAP稳定性表原始变量完整", len(table) == 25 and table["变量"].is_unique
                and set(table["变量"]) == set(RAW_FEATURES), "25个原始变量")
    ranks = detail.sort_values("重拟合次数")[RAW_FEATURES].to_numpy(dtype=float)
    audit.check("每次SHAP排序严格覆盖1至25", ranks.shape == (STABILITY, 25)
                and np.array_equal(np.sort(ranks, axis=1), np.tile(np.arange(1, 26), (STABILITY, 1))),
                "边界并列按预定规则处理，不能超额入选")
    table = table.set_index("变量").reindex(RAW_FEATURES)
    expected = np.column_stack(((ranks <= 5).mean(axis=0), (ranks <= 10).mean(axis=0), ranks.mean(axis=0),
                                np.median(ranks, axis=0), np.quantile(ranks, .025, axis=0), np.quantile(ranks, .975, axis=0)))
    columns = ["前5入选频率", "前10入选频率", "平均名次", "名次中位数", "名次2.5%分位数", "名次97.5%分位数"]
    audit.close("100次SHAP稳定性频率及名次统计独立复算", table[columns].to_numpy(), expected)
    audit.close("前5及前10入选频率严格守恒", [table["前5入选频率"].sum(), table["前10入选频率"].sum()], [5, 10], atol=1e-12, rtol=0)
    audit.numeric("SHAP稳定性汇总无无穷数", table)


def pdf_dimensions(path):
    """读取Matplotlib单页PDF的MediaBox，不依赖额外PDF运行环境。"""
    data = Path(path).read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("PDF头或结束标记无效")
    if len(re.findall(rb"/Type\s*/Page\b", data)) != 1:
        raise ValueError("图形PDF应只有一页")
    match = re.search(rb"/MediaBox\s*\[\s*([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s*\]", data)
    if match is None:
        raise ValueError("无法读取PDF MediaBox")
    x0, y0, x1, y1 = [float(value) for value in match.groups()]
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("PDF页面尺寸无效")
    return x1 - x0, y1 - y0


def audit_figures(audit):
    folder = audit.out / "图片"
    pngs, pdfs = list(folder.glob("*.png")), list(folder.glob("*.pdf"))
    audit.check("11幅PNG及11幅PDF文件名完整", len(pngs) == len(pdfs) == 11
                and {p.stem for p in pngs} == {p.stem for p in pdfs} == set(FIGURES), "逐图对应，不能以旧文件补足数量")
    for name in FIGURES:
        def one(name=name):
            png, pdf = folder / f"{name}.png", folder / f"{name}.pdf"
            with Image.open(png) as figure:
                figure.verify()
            with Image.open(png) as figure:
                width, height = figure.size
                dpi = figure.info.get("dpi", (0, 0))
                audit.check(f"{name}/PNG可读且600dpi", min(width, height) > 1000
                            and len(dpi) >= 2 and all(abs(float(value) - 600) <= .5 for value in dpi[:2]),
                            f"尺寸{width}×{height}；DPI={tuple(round(float(v), 3) for v in dpi[:2])}")
            pdf_width, pdf_height = pdf_dimensions(pdf)
            audit.check(f"{name}/PDF单页且尺寸对应PNG", abs(pdf_width / 72 * 600 - width) <= 24
                        and abs(pdf_height / 72 * 600 - height) <= 24 and pdf.stat().st_size > 1000,
                        f"PDF页面{pdf_width:.3f}×{pdf_height:.3f} pt")
        audit.section(f"{name}文件核验异常", one)


def main():
    parser = argparse.ArgumentParser(description="不复用原指标函数的独立输出复核")
    parser.add_argument("--结果目录", required=True)
    out = Path(parser.parse_args().结果目录).resolve()
    if not out.is_dir():
        raise FileNotFoundError("结果目录不存在")
    audit = Audit(out)
    metadata = audit.section("研究配置与指纹核验异常", lambda: audit_config(audit))

    def split_target():
        split = load_csv(out / "私密记录_禁止公开" / "固定数据划分.csv")
        require_columns(split, ["记录编号", "标签", "子集"])
        audit.check("固定划分记录唯一及三集完整", split["记录编号"].is_unique
                    and split["记录编号"].notna().all() and set(split["子集"]) == {"训练集", "开发验证集", "最终测试集"}
                    and split["标签"].isin([0, 1]).all(), "每条记录仅属于一个子集")
        n, remainder = len(split), int(math.ceil(.4 * len(split)))
        expected = [n - remainder, remainder - int(math.ceil(.5 * remainder)), int(math.ceil(.5 * remainder))]
        counts = [int(split["子集"].eq(name).sum()) for name in ["训练集", "开发验证集", "最终测试集"]]
        audit.check("固定划分人数符合60/20/20整数规则", counts == expected, str(counts))
        target = split.loc[split["子集"].eq("最终测试集")].set_index("记录编号")["标签"]
        audit.check("固定测试标签两类齐全", set(target.unique()) == {0, 1}, "独立AUC及配对Bootstrap的必要条件")
        return target

    target = audit.section("固定划分核验异常", split_target)
    primary = audit.section("主分析预测文件核验异常", lambda: audit_predictions(audit, "主分析", EXPECTED_KEYS, target)) if target is not None else None
    ci = audit.section("主分析置信区间文件读取异常", lambda: load_csv(out / "表格" / "主分析_指标置信区间.csv"))
    effect = audit.section("采样配对效应文件读取异常", lambda: load_csv(out / "表格" / "SMOTEENC相对未采样的配对效应.csv"))
    if primary is not None and ci is not None and effect is not None:
        audit.section("主分析Bootstrap独立复核异常", lambda: audit_bootstrap(audit, primary, ci, effect))
    if metadata is not None and primary is not None:
        frozen = metadata[1]
        code = frozen["主模型键"].split("|")[0]
        reduced = audit.section("去自伤预测文件核验异常", lambda: audit_predictions(
            audit, "去除自伤", [f"{code}|{strategy}" for strategy in STRATEGIES], target, primary["ids"]))
        if reduced is not None:
            audit.section("去自伤Bootstrap独立复核异常", lambda: audit_sensitivity(audit, primary, reduced, frozen))
        audit.section("DeLong独立核验异常", lambda: audit_delong(audit, primary, frozen))
        audit.section("完整校准尺度SHAP核验异常", lambda: audit_shap(audit, primary, frozen))
    audit.section("SHAP稳定性独立复核异常", lambda: audit_stability(audit))
    audit.section("图形文件独立复核异常", lambda: audit_figures(audit))
    audit.finish()


if __name__ == "__main__":
    main()
