"""中文期刊图、表和运行核查报告。图内不设置文章式标题。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.patches import FancyBboxPatch, Polygon
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from 算法与统计 import 模型名称, 策略名称, 连续变量, 分类变量, 指标名称

配色 = {"LR": "#222222", "DT": "#d62728", "RF": "#1f77b4", "ET": "#2ca02c", "XGB": "#9467bd", "LGBM": "#ff7f0e"}
线型 = {"LR": "-", "DT": "--", "RF": "-.", "ET": ":", "XGB": (0, (5, 2, 1, 2)), "LGBM": (0, (3, 1))}
标记 = {"LR": "o", "DT": "s", "RF": "^", "ET": "D", "XGB": "P", "LGBM": "v"}
正色, 负色 = "#ff0051", "#008bfb"
色图 = colors.LinearSegmentedColormap.from_list("SHAP蓝红", [负色, "#9855a3", 正色])


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"] if f in installed), None)
    if family is None:
        raise RuntimeError("没有可用的中文字体；请先安装微软雅黑、黑体或思源黑体后重新出图。")
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "axes.linewidth": .8,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "white"})
    return family


def save(fig, out, name):
    folder = out / "图片"
    # PDF保留矢量文字与曲线；PNG提供600 dpi投稿位图。
    fig.canvas.draw()
    fig.savefig(folder / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=.08)
    fig.savefig(folder / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def clean_ax(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=.6, zorder=0)
    ax.set_axisbelow(True)


def flow(out, audit, d, train, dev, test):
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    def box(x, y, w, h, text, color="#eef3f8"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=.01",
                              linewidth=.8, edgecolor="#6c8093", facecolor=color)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10, linespacing=1.55)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#536878", "lw": 1.2})
    read = int(audit["读取行数"].sum())
    excluded = int(audit["排除全空行"].sum())
    box(.25, .85, .5, .12, f"2021—2025年系统导出记录合并\n读取 {read:,} 条；排除全空行 {excluded:,} 条")
    box(.25, .64, .5, .13, f"纳入 {len(d):,} 条；系统标签阳性 {int(d['标签'].sum()):,} 条\n21个心理维度＋4个人口学变量\n去除直接自杀维度及来源分级等泄漏信息")
    arrow((.5, .85), (.5, .77))
    for x, ids, name, text in [(.015, train, "训练集", "五折交叉验证与参数搜索\n折内插补、标准化、采样和独热编码"),
                              (.355, dev, "开发验证集", "保持原始类别比例\nSigmoid校准、阈值确定与模型选择"),
                              (.695, test, "最终测试集", "保持原始类别比例\n冻结方案后一次性评价与解释")]:
        box(x, .36, .29, .18, f"{name}  n={len(ids):,}\n标签阳性 {int(d.iloc[ids]['标签'].sum()):,} 条\n{text}")
        arrow((.5, .64), (x+.145, .54))
    box(.15, .07, .7, .18, "六类模型 × 未采样／SMOTE-ENC\n区分度、识别表现、概率校准、复核负担与配对不确定性\n校准后log-odds的SHAP解释、重拟合稳定性及去除自伤敏感性分析", "#f4f5ee")
    for x in [.16, .5, .84]: arrow((x, .36), (.5, .25))
    save(fig, out, "图01_数据纳入与实验流程")


def heatmap(out, spearman):
    fig, ax = plt.subplots(figsize=(11, 9.2))
    a = spearman.to_numpy()
    im = ax.imshow(a, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="none")
    ax.set_xticks(range(len(spearman)), spearman.columns, rotation=58, ha="right", fontsize=8)
    ax.set_yticks(range(len(spearman)), spearman.index, fontsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for i in range(len(a)):
        for j in range(len(a)):
            ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center", fontsize=6.1,
                    color="white" if abs(a[i,j]) > .65 else "#303030")
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cb.set_label("Spearman相关系数")
    fig.tight_layout()
    save(fig, out, "图02_心理维度Spearman相关热图")


def bar_comparison(out, table, ci, metric, name, ylabel):
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    t = table.set_index("键")
    for s, strategy in enumerate(策略名称):
        xs, values, lower, upper = [], [], [], []
        for i, code in enumerate(模型名称):
            key = f"{code}|{strategy}"
            c = ci[(ci["键"] == key) & (ci["指标"] == 指标名称[metric])].iloc[0]
            value = t.loc[key, metric]
            x = i + (s - .5) * .34
            ax.bar(x, value, width=.31, color=配色[code], alpha=1. if s else .35,
                   edgecolor=配色[code], hatch="" if s else "//", linewidth=.8, zorder=2)
            xs.append(x); values.append(value)
            lower.append(max(0., value-c["95%下限"])); upper.append(max(0., c["95%上限"]-value))
            ax.text(x, min(1.065, max(value, c["95%上限"])+.018), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=0)
        ax.errorbar(xs, values, yerr=[lower, upper], fmt="none", color="#303030", linewidth=.8, capsize=2.4, zorder=3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777777", alpha=.35, hatch="//", label="未采样"),
                       Patch(facecolor="#777777", label="SMOTE-ENC")], ncol=2, loc="upper right")
    ax.set_xticks(range(6), list(模型名称.values()))
    ax.set_ylim(0, 1.12); ax.set_ylabel(ylabel)
    ax.set_yticks(np.linspace(0, 1, 6))
    clean_ax(ax)
    fig.tight_layout()
    save(fig, out, name)


def discrimination(out, probs, y, is_pr=False):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5), sharex=True, sharey=True)
    for ax, strategy, letter in zip(axes, 策略名称, "ab"):
        for code, model in 模型名称.items():
            p = probs[f"{code}|{strategy}"]
            if is_pr:
                precision, recall, _ = precision_recall_curve(y, p)
                from sklearn.metrics import average_precision_score
                ax.plot(recall, precision, color=配色[code], lw=1.45,
                        linestyle=线型[code], marker=标记[code], markersize=3, markerfacecolor="white",
                        markevery=max(1, len(recall)//10), label=f"{model} (AP={average_precision_score(y, p):.3f})")
            else:
                fpr, tpr, _ = roc_curve(y, p)
                from sklearn.metrics import roc_auc_score
                ax.plot(fpr, tpr, color=配色[code], lw=1.45,
                        linestyle=线型[code], marker=标记[code], markersize=3, markerfacecolor="white",
                        markevery=max(1, len(fpr)//10), label=f"{model} (ROC-AUC={roc_auc_score(y, p):.3f})")
        if is_pr:
            ax.axhline(np.mean(y), color="#777777", linestyle="--", lw=1., label=f"标签阳性率={np.mean(y):.3f}")
            ax.set_xlabel("召回率")
            ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor="white", framealpha=.95)
        else:
            ax.plot([0, 1], [0, 1], "--", color="#999999", lw=.8)
            ax.set_xlabel("1−特异度")
            ax.legend(loc="lower right", fontsize=8)
        ax.text(0, 1.025, f"({letter}) {strategy}", transform=ax.transAxes, va="bottom")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.015)
        ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("精确率" if is_pr else "召回率")
    fig.tight_layout()
    save(fig, out, "图05_六模型PR曲线" if is_pr else "图04_六模型ROC曲线")


def calibration(out, bins, code):
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    a = bins[bins["模型"] == 模型名称[code]]
    diagnostics = pd.read_csv(out / "表格" / "主分析_概率校准诊断.csv", float_precision="round_trip")
    for strategy, color in [("未采样", "#1f77b4"), ("SMOTE-ENC", "#d62728")]:
        for phase, linestyle, marker in [("校准前", "--", "o"), ("校准后", "-", "s")]:
            b = a[(a["策略"] == strategy) & (a["阶段"] == phase)]
            diagnostic = diagnostics[(diagnostics["模型"] == 模型名称[code]) &
                                      (diagnostics["策略"] == strategy) & (diagnostics["阶段"] == phase)]
            if len(diagnostic) != 1:
                raise ValueError(f"校准诊断记录不唯一或缺失：{code}/{strategy}/{phase}")
            brier = float(diagnostic.iloc[0]["Brier分数"])
            ax.errorbar(b["平均预测概率"], b["观察阳性比例"],
                        yerr=[b["观察阳性比例"]-b["观察比例95%下限"], b["观察比例95%上限"]-b["观察阳性比例"]],
                        fmt=marker, markersize=3.5, linestyle=linestyle, color=color,
                        alpha=1. if phase == "校准后" else .6, lw=1, capsize=1.7,
                        label=f"{strategy}，{phase} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], ":", color="#666666", lw=1., label="理想校准线")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="平均预测概率", ylabel="系统标签观察阳性比例")
    handles, labels = ax.get_legend_handles_labels()
    # errorbar的句柄类型与普通折线不同，显式固定图例次序。
    order = [i for i, label in enumerate(labels) if label != "理想校准线"] + [labels.index("理想校准线")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", fontsize=8)
    fig.text(.5, .018, "Brier分数：整体概率预测误差，越小越好", ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, out, "图07_主模型概率校准曲线")


def matrices(out, fitted, probs, y, code):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    matrices = [confusion_matrix(y, probs[f"{code}|{s}"] >= fitted[f"{code}|{s}"].threshold, labels=[0, 1]) for s in 策略名称]
    vmax = max(x.max() for x in matrices)
    for ax, matrix, strategy, letter in zip(axes, matrices, 策略名称, "AB"):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18,
                        color="white" if matrix[i,j] > vmax/2 else "#222222")
        ax.set_xticks([0, 1], ["预测阴性", "预测阳性"])
        ax.set_yticks([0, 1], ["标签阴性", "标签阳性"])
        ax.set_xlabel("模型筛查分类")
        ax.text(0, 1.06, f"({letter}) {strategy}", transform=ax.transAxes)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("系统标签")
    fig.subplots_adjust(wspace=.32, right=.87, bottom=.16, top=.88)
    cax = fig.add_axes([.9, .18, .02, .66])
    fig.colorbar(im, cax=cax, label="记录数")
    save(fig, out, "图08_主模型混淆矩阵")


def beeswarm(out, values, names, X):
    order = np.argsort(np.mean(np.abs(values), axis=0))[::-1][:10]
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.7, 5.8))
    for row, j in enumerate(order):
        v = values[:, j]
        # 在每个横坐标分箱内对称堆叠；堆叠上限固定，点的位置不改变SHAP值。
        edges = np.linspace(v.min()-1e-9, v.max()+1e-9, 65)
        bin_id = np.clip(np.digitize(v, edges)-1, 0, 63)
        offsets = np.zeros(len(v))
        max_bin = max(np.sum(bin_id == b) for b in np.unique(bin_id))
        for b in np.unique(bin_id):
            ids = np.flatnonzero(bin_id == b)
            rng.shuffle(ids)
            levels = np.array([(-1 if i % 2 else 1) * ((i+1)//2) for i in range(len(ids))])
            offsets[ids] = levels * .67 / max(1, max_bin)
        if names[j] in 连续变量:
            raw = pd.to_numeric(X[names[j]], errors="coerce").to_numpy()
            low, high = np.nanquantile(raw, [.05, .95])
            normed = np.clip((raw-low)/max(1e-9, high-low), 0, 1)
            normed = np.nan_to_num(normed, nan=.5)
            ax.scatter(v, row+offsets, c=normed, cmap=色图, vmin=0, vmax=1, s=7, alpha=.72, linewidths=0, rasterized=True)
        else:
            ax.scatter(v, row+offsets, color="#888888", s=7, alpha=.72, linewidths=0, rasterized=True)
        ax.axhline(row, color="#e8e8e8", lw=.5, zorder=0)
    ax.axvline(0, color="#777777", lw=.7, zorder=0)
    ax.set_yticks(range(10), [names[j] for j in order])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP值（校准后log-odds）")
    # 仅调整显示范围，不截断、缩放或改写原始贡献值。
    xmin = min(-.5, float(np.floor(np.min(values[:, order]) * 2) / 2))
    ticks = np.arange(xmin, 3.01, .5)
    ax.set_xlim(xmin, 3)
    ax.set_xticks(ticks, [f"+{v:g}" if v > 0 else f"{v:g}" for v in ticks])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=色图), ax=ax, fraction=.025, pad=.04)
    cb.set_ticks([0, 1], labels=["低", "高"]); cb.set_label("特征贡献值")
    fig.tight_layout()
    save(fig, out, "图09_主模型SHAP蜂群图_前十变量")


def waterfalls(out, values, base, names, choice, X):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    for ax, case, letter, case_label in zip(axes, choice, "ab", ["预测阳性案例", "预测阴性案例"]):
        phi = values[case]
        top = np.argsort(np.abs(phi))[::-1][:5]
        others = [j for j in range(len(phi)) if j not in top]
        contributions = np.r_[phi[others].sum(), phi[top[::-1]]]
        labels = ["其他特征"]
        for j in top[::-1]:
            raw = X.iloc[case][names[j]]
            raw_text = f"{raw:.2f}" if names[j] in 连续变量 and pd.notna(raw) else str(raw)
            labels.append(f"{names[j]} = {raw_text}")
        current = float(base[case])
        positions = [current]
        span = max(.1, float(np.ptp(np.r_[current, current+np.cumsum(contributions)])))
        for i, value in enumerate(contributions):
            end = current + value
            direction = 1 if value >= 0 else -1
            head = min(abs(value) * .35, span * .045)
            shoulder = end - direction * head
            vertices = [(current, i-.29), (shoulder, i-.29), (end, i),
                        (shoulder, i+.29), (current, i+.29)]
            ax.add_patch(Polygon(vertices, closed=True, facecolor=正色 if value >= 0 else 负色,
                                 edgecolor="none", alpha=.95, zorder=2))
            ax.axhline(i, color="#e8e8e8", lw=.6, linestyle=":", zorder=0)
            if abs(value) < span * .15:
                offset = .025 * span * (1 if value >= 0 else -1)
                ax.text(current+value+offset, i, f"{value:+.2f}", ha="left" if value >= 0 else "right",
                        va="center", color="#303030", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .5, "alpha": .9})
            else:
                ax.text(current+value/2, i, f"{value:+.2f}", ha="center", va="center", color="white", fontsize=8)
            if i < len(contributions)-1:
                ax.plot([end, end], [i+.29, i+.71], color="#d0d0d0", lw=.6, linestyle="--", zorder=1)
            current += value
            positions.append(current)
        ax.axvline(base[case], color="#cccccc", lw=.65, linestyle=":", zorder=0)
        ax.axvline(current, color="#c7c7c7", lw=.65, linestyle="--", zorder=0)
        ax.set_yticks(range(6), labels, fontsize=8)
        ax.set_ylim(-.65, 6.05)
        low, high = min(positions), max(positions)
        pad = max(.8, (high-low)*.15)
        ax.set_xlim(low-pad, high+pad)
        ax.text(base[case], -.53, rf"$E[f(x)]={base[case]:.2f}$", ha="center", va="center", fontsize=9)
        ax.text(current, 5.65, rf"$f(x)={current:.2f}$", ha="center", va="center", fontsize=9)
        ax.set_xlabel("校准后log-odds")
        ax.text(0, 1.05, f"({letter}) {case_label}", transform=ax.transAxes)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.19, right=.99, bottom=.16, top=.86, wspace=.72)
    save(fig, out, "图10_主模型SHAP瀑布图_前五变量")


def stability_plot(out, table):
    a = table.head(10).iloc[::-1]
    y = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    ax.barh(y-.17, a["前5入选频率"], height=.3, color="#d62728", label="前5入选频率")
    ax.barh(y+.17, a["前10入选频率"], height=.3, color="#1f77b4", label="前10入选频率")
    ax.set_yticks(y, a["变量"])
    ax.set_xlim(0, 1.17)
    ax.set_xticks(np.linspace(0, 1, 6))
    for i, (_, row) in enumerate(a.iterrows()):
        ax.text(1.02, i, f"{row['名次中位数']:g}", va="center", fontsize=9)
    ax.text(1.02, len(a)-.35, "名次中位数", va="bottom", ha="center", fontsize=8)
    ax.set_xlabel("全流程重拟合中的入选比例")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, out, "图11_SHAP重要性重拟合稳定性")


def create_all_figures(out, d, audit, train, dev, test, spearman, fitted, probs, table, ci, bins, selected, shap_results):
    font = style()
    code = selected.split("|")[0]
    y = d.loc[test, "标签"].to_numpy()
    values, base, importance, choice, stability, info = shap_results
    flow(out, audit, d, train, dev, test)
    heatmap(out, spearman)
    bar_comparison(out, table, ci, "Recall", "图03_六模型召回率对比", "召回率")
    discrimination(out, probs, y)
    discrimination(out, probs, y, True)
    bar_comparison(out, table, ci, "Specificity", "图06_六模型特异度对比", "特异度")
    calibration(out, bins, code)
    matrices(out, fitted, probs, y, code)
    beeswarm(out, values, fitted[selected].preprocessor.raw_names_, d.iloc[test])
    waterfalls(out, values, base, fitted[selected].preprocessor.raw_names_, choice, d.iloc[test])
    stability_plot(out, stability)
    export_workbook(out, table, ci)
    captions = """# 图注与展示口径

所有图均来自同一次固定划分的完整实验。PDF为矢量版，PNG为600 dpi版；图号和图题由论文排版时放在图外。

图01：数据纳入与实验流程。五个年份合并后随机分层划分，工作表风险分级及来源信息均不进入预测变量。

图02：21个心理维度在训练集中的Spearman相关结构，使用成对非缺失观测；相关分析不用于筛选预测变量。

图03、图06：十二组冻结模型在同一测试集中的召回率及特异度；误差线为2000次分层配对Bootstrap的百分位95%区间。所有模型使用开发集固定阈值，未按测试结果重新调整。

图04、图05：两种采样策略下六模型的ROC与PR曲线。图例分别列出ROC-AUC与平均精确率AP，均保留三位小数；AP不与梯形积分PR-AUC混用。PR图虚线为测试集系统标签阳性率。

图07：开发集拟合Sigmoid校准器后，在测试集比较两种采样策略的校准前后预测。分位分组最多10组，重复分位点合并；误差线为组内观察阳性比例的Wilson 95%区间，不是模型参数不确定性区间。灰色对角线为理想校准线。图例报告各策略校准前后的测试集Brier分数（四位小数）；Brier分数衡量整体概率预测误差，越小越好，同时受校准程度和区分能力等因素影响，并非单独衡量校准程度的指标。详细分组坐标与区间见《主分析_校准曲线分组与区间.csv》，校准截距和斜率见《主分析_概率校准诊断.csv》。

图08：主模型家族两种采样策略的混淆矩阵；阈值均在开发集独立确定，色标一致。

图09：主模型前10个原始变量的SHAP蜂群图。归因对象为最终校准后log-odds，正贡献增加模型预测倾向，负贡献降低预测倾向；连续变量以蓝到红表示取值由低到高，分类变量若进入前10则用灰色，避免赋予无序类别高低含义。分类变量的独热贡献先按原始变量求和。横向位置是实际SHAP贡献值，纵向堆叠仅用于展示密度。横坐标刻度间隔为0.5，显示上限为+3，超出显示范围的点不展示，原始贡献值完整保留。色条按要求标为“特征贡献值”，颜色编码仍表示连续特征的取值高低。

图10：按固定种子分别从预测阳性与预测阴性中随机抽取一例，不按标签是否预测正确筛选。各例仅展示绝对贡献最大的5个变量，其余贡献合并；红色向右箭头表示正贡献，蓝色向左箭头表示负贡献。E[f(x)]为基线，f(x)为该例校准后的log-odds，基线加各贡献等于模型输出。案例标签、概率和阈值见《瀑布图案例说明.csv》，图中不放可识别身份的信息。

图11：固定参数后100次全流程分层重拟合的解释稳定性。训练与开发记录在各自角色内有放回重采样，重新执行预处理、重采样、分类器拟合和概率校准；解释背景和开发解释样本固定。展示原主模型前10个变量的前5/前10入选频率，右侧为重拟合名次中位数。

所有阳性、阴性及漏识统计均以现有系统量表标签为参照。本研究没有独立真实意念标签或后续自杀事件结局，图表不能证实模型识别了隐瞒意念的学生。SHAP关联可作为进一步复核及干预研究的候选线索，不能解释为已验证的因果干预靶点。
"""
    cases = pd.read_csv(out / "表格" / "瀑布图案例说明.csv")
    captions += "\n图10案例数值：" + "；".join(
        f"{r['案例']}：系统标签={int(r['系统标签'])}，模型概率={r['模型概率']:.4f}，固定阈值={r['固定阈值']:.4f}"
        for _, r in cases.iterrows()) + "。\n"
    (out / "图注与展示口径.md").write_text(captions, encoding="utf-8")
    (out / "复现记录" / "图形字体与格式.json").write_text(json.dumps({"中文字体": font, "PNG分辨率": 600,
        "矢量格式": "PDF，TrueType字体嵌入", "图数": 11, "图内文章标题": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def export_workbook(out, table, ci):
    """只汇总组水平结果；不将逐记录预测、索引或SHAP放进阅读版工作簿。"""
    paper_rows = []
    for _, row in table.iterrows():
        item = {"模型": row["模型"], "采样策略": row["策略"]}
        for name in ["AUC", "AP", "Recall", "Specificity", "PPV", "NPV", "F1", "Brier", "Review1000", "Miss1000"]:
            a = ci[(ci["键"] == row["键"]) & (ci["指标"] == 指标名称[name])].iloc[0]
            digits = 1 if name.endswith("1000") else 3
            item[指标名称[name]+"（95%区间）"] = f"{row[name]:.{digits}f} ({a['95%下限']:.{digits}f}, {a['95%上限']:.{digits}f})"
        paper_rows.append(item)
    pd.DataFrame(paper_rows).to_csv(out / "表格" / "论文主结果表_估计值与区间.csv", index=False, encoding="utf-8-sig")
    import subprocess, os, sys, shutil
    frames = []
    chosen = ["论文主结果表_估计值与区间", "主分析_测试集完整指标", "主分析_指标置信区间", "SMOTEENC相对未采样的配对效应", "去除自伤前后测试表现比较", "样本构成与固定划分", "变量缺失与未知类别", "主分析_概率校准诊断", "主模型DeLong检验与Holm校正", "主模型全部变量SHAP重要性", "SHAP全流程重拟合稳定性", "训练集点二列相关与校正", "训练集方差膨胀因子"]
    for name in chosen:
        frame = pd.read_csv(out / "表格" / (name+".csv"), float_precision="round_trip")
        frame = frame.rename(columns={c: c.replace("人数", "记录数") for c in frame.columns})
        values = [list(frame.columns)] + frame.astype(object).where(pd.notna(frame), None).values.tolist()
        frames.append({"name": name[:31], "values": values})
    (out / "复现记录" / "工作簿表格输入.json").write_text(json.dumps(frames, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    repo = next(p for p in Path(__file__).resolve().parents if (p / "code").is_dir() and (p / "data").is_dir())
    node = os.environ.get("CODEX_ARTIFACT_NODE") or shutil.which("node")
    backend = os.environ.get("WORKBOOK_EXPORT_BACKEND", "auto")
    if backend not in {"auto", "artifact", "openpyxl"}:
        raise ValueError("WORKBOOK_EXPORT_BACKEND must be auto, artifact or openpyxl")
    if backend == "artifact" and not node:
        raise RuntimeError("The artifact workbook backend requires Node.js")
    if node and backend != "openpyxl":
        env = dict(os.environ, ANALYSIS_PYTHON=sys.executable)
        subprocess.run([str(node), str(repo / "code" / "zh" / "工作簿导出.mjs"), str(out)], check=True, env=env)
    else:
        subprocess.run([sys.executable, str(repo / "code" / "en" / "export_workbook.py"), str(out),
                        "--input", str(out / "复现记录" / "工作簿表格输入.json"),
                        "--filename", "论文结果汇总.xlsx", "--record-dir", "复现记录"], check=True)


def write_report(out, table, ci, effect, selection, fitted, shap_info, sensitivity, n, ntest, elapsed):
    table = table.copy()
    selected = selection["主模型键"]
    row = table.set_index("键").loc[selected]
    below = table[table["Recall"] < .88]
    perfect = table[table["Recall"] >= 1-1e-12]
    checks = []
    def check(name, success, detail):
        checks.append({"检查项": name, "通过": bool(success), "说明": detail})
    check("主分析十二组结果齐全", len(table) == 12 and table["键"].nunique() == 12, "六模型×两种采样策略")
    check("混淆矩阵人数一致", bool(((table["TP"]+table["FP"]+table["TN"]+table["FN"]) == ntest).all()), f"测试人数={ntest}")
    check("召回率与混淆矩阵一致", bool(np.allclose(table["Recall"], table["TP"]/(table["TP"]+table["FN"]))), "不隐藏或手动修改数值")
    check("特异度与混淆矩阵一致", bool(np.allclose(table["Specificity"], table["TN"]/(table["TN"]+table["FP"]))), "同一阈值计算")
    check("主指标没有无穷值", not bool(np.isinf(table[list(指标名称)].to_numpy()).any()), "零分母按NaN不可估计保存")
    check("每组指标区间齐全", len(ci) == 12*len(指标名称), f"实际{len(ci)}行")
    check("采样配对效应齐全", len(effect) == 6*len(指标名称), f"实际{len(effect)}行")
    check("SHAP可加性", shap_info["最大可加性残差"] < 1e-5, f"最大残差={shap_info['最大可加性残差']:.3g}")
    check("敏感性分析两种策略齐全", len(sensitivity) == 2, "去除自伤后独立重新调参")
    for ext in ["png", "pdf"]:
        files = list((out / "图片").glob(f"*.{ext}"))
        check(f"11幅{ext.upper()}图齐全且非空", len(files) == 11 and all(p.stat().st_size > 1000 for p in files), f"实际{len(files)}幅")
    pd.DataFrame(checks).to_csv(out / "输出核查清单.csv", index=False, encoding="utf-8-sig")
    if not all(c["通过"] for c in checks):
        raise AssertionError("输出核查未全部通过，见输出核查清单.csv。")
    numeric = table[["模型", "策略", "AUC", "AP", "Recall", "Specificity", "PPV", "Brier", "FP", "FN"]].rename(columns=指标名称)
    def markdown(df):
        # 避免依赖额外的tabulate包。
        columns = list(df.columns)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
        for values in df.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
        return "\n".join(lines)
    below_text = "；".join(f"{r.模型}/{r.策略}={r.Recall:.3f}" for r in below.itertuples()) or "无"
    perfect_text = "；".join(f"{r.模型}/{r.策略}" for r in perfect.itertuples()) or "无"
    code = selected.split("|")[0]
    comparison = effect[(effect["模型"] == 模型名称[code]) & effect["指标"].isin(["平均精确率AP", "召回率", "特异度", "Brier分数", "每千条记录复核数"])]
    continuation = "绘图接口修复后使用已冻结结果接续生成图表，原始建模源码及数值结果校验值分别存档；未重新训练、选模或改动阈值。" if (out / "复现记录" / "后处理源码版本.json").exists() else ""
    text = f"""# 完整实验运行与结果核查

本轮纳入{n}条记录，固定测试集{ntest}条。完整运行约{elapsed/60:.1f}分钟（含断点复用时仅表示本次启动耗时）。十二组模型、2000次分层配对Bootstrap、100次解释重拟合及去除自伤敏感性分析已完成。

{continuation}

开发阶段预先选定的主模型为**{selection['主模型']}／{selection['主策略']}**。最终测试ROC-AUC={row['AUC']:.4f}，AP={row['AP']:.4f}，召回率={row['Recall']:.4f}，特异度={row['Specificity']:.4f}，Brier={row['Brier']:.4f}。模型选择发生在测试集评价之前，冻结记录见《测试集启封前的冻结方案.json》。

{markdown(numeric)}

测试召回率低于0.88的组：{below_text}。测试召回率等于1的组：{perfect_text}。上述结果均完整保留，没有通过测试集调阈值、改种子或删模型改变图表。开发阶段目标不能保证在独立测试样本上同样达到。

主模型家族的采样效应如下，差值方向为SMOTE-ENC减未采样。是否支持重采样应结合差值区间、实际敏感度和复核负担判断；区间跨零只表示当前数据证据不足，不能证明两种方法等效或心理数据普遍无需采样。

{markdown(comparison)}

程序核查项目全部通过，见《输出核查清单.csv》。另需结合图片视觉核查字体、标注和版面。所有预测及SHAP逐记录数据、划分索引、拟合模型和运行断点放在《私密记录_禁止公开》，不应上传公开仓库。

研究结局是系统量表标记的自杀意念，而非已独立确认的真实意念或未来自杀行为。模型预测阳性而系统标签阴性的记录只能视为待复核候选，现有数据不能验证其为隐瞒意念者。SHAP定义为对最终校准后log-odds的可加归因，最大可加性残差{shap_info['最大可加性残差']:.3g}；归因关联不等于因果干预证据。循证医学可在后续论文引言与讨论中作为审慎的研究框架提及。

本轮仅生成中文代码、结果及运行记录，未修改论文或生成投稿补充Word。是否选用哪些图表及如何讨论，需以本轮全部真实结果为依据。
"""
    (out / "结果检验报告.md").write_text(text, encoding="utf-8")
    return f"核查通过：{len(checks)}项；主模型{selection['主模型']}/{selection['主策略']}，测试召回率{row['Recall']:.3f}、特异度{row['Specificity']:.3f}。"
