"""验证会实质影响研究结论的算法边界，不接触真实受测者数据。"""
import sys
from pathlib import Path
目录 = Path(__file__).resolve().parent
sys.path.insert(0, str(目录 / "依赖环境"))
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from 算法与统计 import *


class ScientificTests(unittest.TestCase):
    def test_original_feature_masking_in_calibrated_tree_shap(self):
        rng = np.random.default_rng(17)
        x = pd.DataFrame({"焦虑": rng.normal(size=180), "抑郁": rng.normal(size=180)})
        for c in 分类变量:
            x[c] = rng.choice(["甲", "乙"], size=len(x))
        y = rng.binomial(1, expit(-1 + x["焦虑"].to_numpy()))
        fit = FittedExperiment("DT", "未采样", {"max_depth": 3, "min_samples_leaf": 6}, numeric=["焦虑", "抑郁"])
        fit.fit(x.iloc[:120], y[:120], x.iloc[120:160], y[120:160])
        phi, base, info = explain_calibrated(fit, x.iloc[160:163], x.iloc[:8], permutations=2)
        self.assertEqual(phi.shape, (3, 6))
        np.testing.assert_allclose(base + phi.sum(axis=1), fit.decision_function(x.iloc[160:163]), atol=1e-10)

    def test_all_missing_training_column_is_blocked(self):
        d = pd.DataFrame({"焦虑": [np.nan, np.nan], **{c: ["未知", "已知"] for c in 分类变量}})
        with self.assertRaises(ValueError):
            MixedPreprocessor(["焦虑"]).fit(d)

    def test_undefined_metrics_are_nan_not_inf(self):
        y = np.array([0, 0, 1, 1])
        p = np.full(4, .1)
        a = metrics(y, p, .5)
        b = fast_metric_array(y, np.array([p]), [.5])[0]
        self.assertTrue(np.isnan(a["PPV"]))
        self.assertTrue(np.isnan(a["ReviewPerPositive"]))
        self.assertFalse(np.isinf(b).any())

    def test_delong_nontrivial_known_covariance(self):
        # 两个模型、三名阳性三名阴性的独立手算结构分量。
        y = np.array([1, 1, 1, 0, 0, 0])
        p = np.array([.9, .5, .3, .8, .4, .2])
        q = np.array([.8, .7, .1, .9, .3, .2])
        v1 = np.array([[1., 2/3, 1/3], [2/3, 2/3, 0.]])
        v0 = np.array([[1/3, 2/3, 1.], [0., 2/3, 2/3]])
        cov = np.cov(v1, ddof=1)/3 + np.cov(v0, ddof=1)/3
        expected = np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])
        result = delong_compare(y, [p, q], ["甲", "乙"], "甲").iloc[0]
        self.assertAlmostEqual(result["AUC差值"], 2/9)
        self.assertAlmostEqual(result["差值标准误"], expected)

    def test_exported_blank_label_semantics_and_missing_column(self):
        from 完整实验 import read_data
        root = 目录 / "运行缓存" / "合成数据标签核查"
        inputs, outputs = root / "输入", root / "输出"
        inputs.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        d = pd.DataFrame({c: [0., 1., 2.] for c in 连续变量}).rename(columns={"幻觉妄想": "幻觉、妄想症状", "就业问题": "就业压力"})
        for c in 分类变量: d[c] = ["未知", "已知", "已知"]
        d["建议关注方向"] = [None, "抑郁、自杀意念", "焦虑"]
        d["自杀意念"] = [99., 0., 99.]  # 与标签故意反向，验证不使用直接维度建Y。
        d["家庭关系满意度"] = np.nan
        d["学校及专业满意度"] = np.nan
        for year in range(2021, 2026): d.to_excel(inputs / f"{year}.xlsx", index=False)
        clean, audit, sources = read_data(inputs, outputs)
        self.assertEqual(clean["标签"].tolist(), [0, 1, 0] * 5)
        self.assertEqual(int(audit["关注方向空白阴性数"].sum()), 5)
        prep = MixedPreprocessor().fit(clean)
        self.assertFalse(set(prep.raw_names_) & set(禁止特征))
        self.assertEqual(len(prep.raw_names_), 25)
        d.drop(columns="建议关注方向").to_excel(inputs / "2021.xlsx", index=False)
        try:
            with self.assertRaises(ValueError): read_data(inputs, outputs)
        finally:
            d.to_excel(inputs / "2021.xlsx", index=False)

    def test_supervised_encoding_and_original_categories(self):
        n = np.arange(20, dtype=float)[:, None]
        c = np.array([["甲"]] * 10 + [["乙"]] * 10)
        y = np.array([1] * 4 + [0] * 6 + [1] * 2 + [0] * 8)
        sampler = SMOTEENC(1., 3)
        nn, cc, yy = sampler.fit_resample(n, c, y)
        scale = np.std(n[y == 1], axis=0, ddof=0)[0]
        self.assertAlmostEqual(sampler.encoding_[0]["甲"], (4 - 3) / 3 * scale)
        self.assertAlmostEqual(sampler.encoding_[0]["乙"], (2 - 3) / 3 * scale)
        self.assertEqual(np.bincount(yy).tolist(), [14, 14])
        np.testing.assert_array_equal(nn[:20], n)
        self.assertEqual(set(cc.ravel()), {"甲", "乙"})
        # 编码相同的类别仍必须完整保留原始标签。
        c2 = np.array([["甲"], ["乙"]] * 10)
        s2 = SMOTEENC(1., 3)
        _, output, _ = s2.fit_resample(n, c2, np.r_[np.ones(6, int), np.zeros(14, int)])
        self.assertAlmostEqual(s2.encoding_[0]["甲"], s2.encoding_[0]["乙"])
        np.testing.assert_array_equal(output[:20], c2)

    def test_train_only_preprocessor(self):
        train = pd.DataFrame({"焦虑": [1., np.nan, 3.], "性别": ["男", "女", "男"],
                              "民族": ["汉族"] * 3, "生源地": ["农村"] * 3, "是否独生": ["否"] * 3})
        unseen = train.iloc[:1].copy()
        unseen["焦虑"], unseen["性别"] = 10000., "未出现类别"
        p = MixedPreprocessor(["焦虑"]).fit(train)
        p.transform(unseen)
        self.assertEqual(p.medians_["焦虑"], 2.)
        self.assertEqual(p.scaler_.mean_[0], 2.)
        self.assertEqual(p.unknown_counts(unseen)["性别"], 1)
        self.assertNotIn("未出现类别", p.encoder_.categories_[0])

    def test_threshold_ties(self):
        y = np.r_[np.ones(10, int), np.zeros(10, int)]
        p = np.r_[[.9, .8, .7, .6, .5, .4, .3, .2, .2, .1], np.linspace(.05, .95, 10)]
        t = select_threshold(y, p)
        self.assertEqual(t, .2)
        self.assertEqual(np.mean(p[y == 1] >= t), .9)
        self.assertLess(np.mean(p[y == 1] > t), .88)

    def test_calibration_and_shap_additivity(self):
        rng = np.random.default_rng(3)
        score = rng.normal(size=500)
        y = rng.binomial(1, expit(-1 + 1.8 * score))
        cal = SigmoidCalibrator().fit(score, y)
        self.assertLess(cal.gradient_, 1e-5)
        coef, intercept = np.array([.4, -.6, 1.2]), -.3
        a, bg = rng.normal(size=(20, 3)), rng.normal(size=(80, 3))
        phi = (a - bg.mean(axis=0)) * (cal.slope * coef)
        base = cal.intercept + cal.slope * (intercept + bg.mean(axis=0) @ coef)
        np.testing.assert_allclose(base + phi.sum(axis=1), cal.decision_function(intercept + a @ coef), atol=1e-12)

    def test_delong_ties_identical(self):
        y = np.r_[np.ones(6), np.zeros(6)]
        p = np.array([.9, .8, .7, .6, .5, .5, .6, .5, .4, .4, .2, .1])
        output = delong_compare(y, [p, p, 1 - p], ["甲", "乙", "丙"], "甲")
        self.assertAlmostEqual(output.iloc[0]["AUC差值"], 0.)
        self.assertEqual(output.iloc[0]["原始P值"], 1.)
        self.assertAlmostEqual(output.iloc[1]["AUC差值"], roc_auc_score(y, p) - roc_auc_score(y, 1-p))
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])

    def test_bootstrap_stratification_and_vector_metrics(self):
        y = np.r_[np.ones(7, int), np.zeros(13, int)]
        p = np.linspace(.99, .01, 20)
        ids = bootstrap_indices(y, np.random.default_rng(3))
        self.assertEqual(np.bincount(y[ids]).tolist(), [13, 7])
        scalar = metrics(y, p, .4)
        vector = fast_metric_array(y, np.array([p]), [.4])[0]
        np.testing.assert_allclose(vector, [scalar[k] for k in 指标名称])


if __name__ == "__main__":
    unittest.main(verbosity=2)
