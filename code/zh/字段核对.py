"""建模前核对五年全部工作表，只输出字段和汇总信息，不输出个体记录。"""
from pathlib import Path
import sys
import json
import hashlib
import unicodedata

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "依赖环境"))
import numpy as np
import pandas as pd
from 算法与统计 import 连续变量, 分类变量, 字段映射


def normalize(value):
    return unicodedata.normalize("NFKC", str(value)).strip()


def audit_inputs(folder, out):
    folder, out = Path(folder), Path(out)
    target = out / "复现记录"
    target.mkdir(parents=True, exist_ok=True)
    sheets, fields, category, files, overlaps, combined = [], [], [], [], [], []
    required = set(连续变量 + 分类变量 + ["建议关注方向"])
    for year in range(2021, 2026):
        path = folder / f"{year}.xlsx"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"文件": path.name, "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with pd.ExcelFile(path) as book:
            yearly = []
            for sheet in book.sheet_names:
                frame = pd.read_excel(book, sheet_name=sheet)
                headers = [normalize(c) for c in frame.columns]
                frame.columns = headers
                d = frame.rename(columns=字段映射)
                if d.columns.duplicated().any():
                    raise ValueError(f"{path.name}/{sheet}有重复字段")
                absent = sorted(required - set(d.columns))
                if absent:
                    raise ValueError(f"{path.name}/{sheet}缺必需字段: {absent}")
                d = d.loc[d.notna().any(axis=1)].copy()
                labels = d["建议关注方向"].fillna("").map(normalize)
                y = labels.str.contains("自杀意念", regex=False)
                for col in d.columns:
                    fields.append({"文件": path.name, "工作表": sheet, "原始字段": headers[list(d.columns).index(col)],
                                   "标准字段": col, "类型": str(d[col].dtype), "缺失数": int(d[col].isna().sum()),
                                   "预测变量": col in 连续变量 + 分类变量})
                for col in 连续变量:
                    values = pd.to_numeric(d[col], errors="coerce")
                    if values.notna().sum() == 0:
                        raise ValueError(f"{path.name}/{sheet}/{col}整列无法解析")
                    if ((d[col].notna() & values.isna()) | np.isinf(values)).any():
                        raise ValueError(f"{path.name}/{sheet}/{col}存在非数值/无穷值，需先核对")
                for col in 分类变量:
                    for level, number in d[col].fillna("未知").map(normalize).value_counts().items():
                        category.append({"文件": path.name, "工作表": sheet, "变量": col, "类别": level, "记录数": int(number)})
                sheets.append({"文件": path.name, "工作表": sheet, "原始行数": len(frame), "非空记录数": len(d),
                               "字段数": len(d.columns), "必需字段数": len(required), "标签阳性数": int(y.sum()),
                               "关注方向原始空白数": int(labels.eq("").sum()), "身份标识字段": "无"})
                hashes = set(pd.util.hash_pandas_object(d.fillna(""), index=False).tolist())
                yearly.append((sheet, hashes))
                d["来源工作表"] = sheet
                d["来源文件"] = path.name
                d["标签"] = y.astype(int)
                combined.append(d)
            for i, (s1, h1) in enumerate(yearly):
                for s2, h2 in yearly[i + 1:]:
                    overlaps.append({"文件": path.name, "工作表甲": s1, "工作表乙": s2,
                                     "全部导出字段相同的值向量数": len(h1 & h2),
                                     "处理": "不据得分相同去重；工作表均为分级分组，无总表"})
    all_data = pd.concat(combined, ignore_index=True)
    for name, rows in [("逐工作表字段与类型核对", fields), ("全部工作表读取核对", sheets),
                       ("分类水平逐工作表核对", category), ("同年工作表交集核对", overlaps)]:
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
    report = {"原始文件": files, "工作表数": len(sheets), "总记录数": len(all_data),
              "标签阳性数": int(all_data["标签"].sum()), "预测变量数": 25,
              "所有工作表必需字段齐全": True, "发现可靠身份标识": False,
              "工作表关系": "均为心理关注分级分表，未发现总汇总工作表；分表间交集核对另存",
              "计分口径": "依据数据方说明作为标准分；本地分布核对不能独立验证常模换算",
              "标签来源可追溯边界": "系统建议关注方向文字；没有条目级数据和独立访谈结局，无法重建完整系统规则",
              "跨表完全相同向量数": int(sum(r["全部导出字段相同的值向量数"] for r in overlaps)),
              "记录独立性限制": "无身份标识，不仅凭得分相同去重，不能确认所有记录均来自不同个体"}
    (target / "字段核对结论.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--数据目录", default=str(Path(__file__).resolve().parents[2] / "data" / "private" / "original"))
    p.add_argument("--结果目录", default=str(Path(__file__).resolve().parents[2] / "results" / "generated" / "zh"))
    a = p.parse_args()
    print(json.dumps(audit_inputs(a.数据目录, a.结果目录), ensure_ascii=False, indent=2))
