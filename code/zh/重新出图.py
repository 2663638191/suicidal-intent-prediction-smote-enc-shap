"""仅从已冻结并保存的绘图输入重绘，不调参、不重新划分、不重新选模。"""
from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "依赖环境"))
import joblib
from 图表输出 import create_all_figures

out = Path(sys.argv[1]) if len(sys.argv) > 1 else CODE.parents[1] / "results" / "generated" / "zh"
if not (out / "私密记录_禁止公开" / "绘图固定输入.joblib").is_file():
    raise SystemExit("重绘需要授权的私密绘图缓存；公开汇总表不能单独重绘个体解释。")
args = list(joblib.load(out / "私密记录_禁止公开" / "绘图固定输入.joblib"))
args[0] = out
create_all_figures(*args)
record = {"时间UTC": datetime.now(timezone.utc).isoformat(),
          "绘图源码SHA256": hashlib.sha256((CODE / "图表输出.py").read_bytes()).hexdigest(),
          "使用输入": "本轮已冻结绘图固定输入.joblib", "重新拟合或开发选择": False}
(out / "复现记录" / "重绘记录.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("已按原冻结预测完成重绘。")
