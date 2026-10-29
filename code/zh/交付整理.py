"""整理代码和字段结构候选副本，不复制结果，不执行网络上传。"""
from pathlib import Path
import argparse
import shutil
import hashlib
import json

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="用于新建候选副本的本地目录")
    args = parser.parse_args()
    destination = args.destination.resolve()
    package = destination / "代码与表头公开候选文件"
    schema = REPO / "data" / "schema"
    if not schema.is_dir():
        raise FileNotFoundError("字段结构目录不存在：" + str(schema))
    if package.exists():
        raise FileExistsError("候选目录已存在；请使用新的目标目录：" + str(package))
    package.mkdir(parents=True)
    shutil.copytree(CODE.parent, package / "code", ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", "运行缓存", "依赖环境", "*.log",
        "私密记录_禁止公开", "private_records_do_not_publish", "*.joblib", "*.pkl",
        "*.pickle", "*.npz", "*.npy", "*.xlsx", "*.xls", "*.csv", "*.png", "*.pdf"))
    shutil.copytree(schema, package / "data" / "schema")
    (package / "README.md").write_text(
        "# 代码与字段结构候选副本\n\n"
        "本目录包含中英文代码和字段结构，运行说明见各代码目录。"
        "原始数据与实证结果未纳入此副本。此命令不执行网络上传。\n", encoding="utf-8")
    files = [{"文件": str(p.relative_to(package)), "SHA256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(package.rglob("*")) if p.is_file()]
    record = destination / "复现记录"
    record.mkdir(parents=True, exist_ok=True)
    (record / "代码与表头候选清单.json").write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"整理{len(files)}份代码与字段文件；未执行上传。")


if __name__ == "__main__":
    main()
