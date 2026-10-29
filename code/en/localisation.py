"""English labels and lossless exports for the frozen screening analysis.

The source field names remain valid input identifiers. Numerical CSV cells are
copied verbatim; only headers and categorical text are localised.
"""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
GLOSSARY = json.loads((HERE / 'resources' / 'frozen_glossary.json').read_text(encoding='utf-8'))
FEATURES = {v['analysis_name_zh']: v['label_en'] for v in GLOSSARY['variables']}
MODELS = {'逻辑回归': 'Logistic regression', '决策树': 'Decision tree',
          '随机森林': 'Random forest', '极端随机树': 'Extremely randomised trees',
          'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM'}
HEADERS = {
 '键':'Model-strategy key','模型':'Model','模型代码':'Model code','策略':'Strategy','采样策略':'Resampling strategy',
 '变量':'Feature','指标':'Metric','ROC-AUC':'ROC-AUC','平均精确率AP':'Average precision',
 '召回率':'Sensitivity','特异度':'Specificity','阳性预测值':'Positive predictive value',
 '阴性预测值':'Negative predictive value','F1值':'F1 score','准确率':'Accuracy',
 '平衡准确率':'Balanced accuracy','Brier分数':'Brier score','筛查阳性比例':'Proportion flagged for review',
 '每千条记录复核数':'Records flagged per 1,000 records',
 '每千条记录漏识标签阳性数':'Missed system-positive records per 1,000 records',
 '每识别一例标签阳性的复核数':'Records reviewed per system-positive record identified',
 '95%下限':'95% CI lower','95%上限':'95% CI upper','估计值':'Estimate',
 '差值_SMOTEENC减未采样':'Difference: SMOTE-ENC minus no resampling',
 '差值95%下限':'Difference 95% CI lower','差值95%上限':'Difference 95% CI upper',
 '有效重复数':'Valid replicates','保留自伤':'Including self-injurious behaviour',
 '去除自伤':'Excluding self-injurious behaviour','去除减保留':'Difference: excluding minus including',
 '分析':'Analysis','子集':'Subset','总人数':'Records','人数':'Records',
 '标签阳性数':'System-positive records','标签阴性数':'System-negative records',
 '阳性人数':'System-positive records','阳性比例':'System-positive proportion','分组':'Group',
 '类别':'Category','比例':'Proportion','非缺失数':'Non-missing records',
 '缺失或未知数':'Missing or unknown records','数量':'Count','中位数':'Median',
 '四分位数下限':'Lower quartile','四分位数上限':'Upper quartile','均值':'Mean',
 '标准差':'Standard deviation','最小值':'Minimum','最大值':'Maximum',
 '平均绝对SHAP':'Mean absolute SHAP value','名次':'Rank','前5入选频率':'Top-five inclusion frequency',
 '前10入选频率':'Top-ten inclusion frequency','平均名次':'Mean rank','名次中位数':'Median rank',
 '名次2.5%分位数':'Rank 2.5th percentile','名次97.5%分位数':'Rank 97.5th percentile',
 '同一记录贡献方向一致率':'Within-record direction agreement',
 '平均绝对SHAP中位数':'Median mean absolute SHAP value','重拟合次数':'Refit number',
 '阶段':'Phase','校准截距':'Calibration intercept','校准斜率':'Calibration slope',
 '平均预测概率':'Mean predicted probability','观察阳性比例':'Observed system-positive proportion',
 '观察比例95%下限':'Observed proportion 95% CI lower','观察比例95%上限':'Observed proportion 95% CI upper',
 '参照模型':'Reference model','对比模型':'Comparison model','AUC差值':'ROC-AUC difference',
 '差值标准误':'Difference standard error','原始P值':'Unadjusted P value','Holm校正P值':'Holm-adjusted P value',
 '点二列相关系数':'Point-biserial correlation','分类器参数':'Classifier parameters',
 '采样比例':'Minority-to-majority ratio','近邻数':'Nearest neighbours',
 '训练阳性数':'Training system-positive records','重采样后阳性数':'System-positive records after resampling',
 '未见类别数':'Unseen categories','有效训练人数':'Training records','分母':'Denominator',
 '样本':'Sample','文件':'File','工作表':'Worksheet','读取行数':'Rows read','排除全空行':'Empty rows excluded',
 '有效行数':'Valid rows','关注方向空白阴性数':'Blank concern field: system-negative records',
 '角色':'Role','处理':'Processing','处理方式':'Processing','问题':'Issue',
 '案例':'Example','系统标签':'System label','模型概率':'Model probability','固定阈值':'Frozen threshold',
}
TEXT = {
 **FEATURES, **MODELS, **HEADERS,
 '未采样':'No resampling','校准前':'Before calibration','校准后':'After calibration',
 '全体':'All records','合并数据':'Combined dataset','训练集':'Training set',
 '开发验证集':'Development validation set','最终测试集':'Final test set',
 '标签阳性':'System positive','标签阴性':'System negative','男':'Male','女':'Female',
 '汉族':'Han Chinese','少数民族':'Ethnic minority','外籍':'Foreign nationality','未知':'Unknown',
 '农村':'Rural area','小城镇':'Small town','中小城市':'Small or medium-sized city','大城市':'Large city',
 '是':'Yes','否':'No','自杀意念':'Suicidal ideation score',
 '建议关注方向':'System-recommended areas of concern','可能问题':'Possible concerns',
 '家庭关系满意度':'Satisfaction with family relationships','学校及专业满意度':'Satisfaction with university and programme',
 '入学年份':'Entry year','来源文件':'Source file','来源工作表':'Source worksheet',
 '源行号':'Source row','记录编号':'Record identifier','标签':'System label',
 '连续预测变量':'Continuous predictor','分类预测变量':'Categorical predictor',
 '禁止进入X':'Excluded from predictors','不作为预测特征':'Not used as a predictor',
 '训练子集内中位数插补及标准化':'Median imputation and standardisation within training subsets',
 '缺失归未知；重采样后独热编码':'Missing levels recorded as Unknown; one-hot encoding after resampling',
 '缺失或待完善统一为未知':'Missing or incomplete category recorded as Unknown',
 '训练中位数插补':'Training-median imputation','训练集中位数插补后':'After training-median imputation',
 '未知类别保留':'Unknown category retained','严重心理危机':'Severe psychological crisis',
 '一般心理问题':'General psychological concerns','潜在心理困扰':'Potential psychological distress',
 '无心理困扰':'No psychological distress',
 '预测阳性案例':'Predicted-positive record','预测阴性案例':'Predicted-negative record',
}

# Source basename -> English basename and a concise Excel worksheet name.
TABLES = {
 '论文主结果表_估计值与区间':('main_results_with_intervals','Main results'),
 '主分析_测试集完整指标':('main_test_metrics','Test metrics'),
 '主分析_指标置信区间':('main_metric_confidence_intervals','Metric confidence intervals'),
 'SMOTEENC相对未采样的配对效应':('paired_resampling_effects','Resampling effects'),
 '去除自伤前后测试表现比较':('paired_self_injury_exclusion_effects','Exclusion effects'),
 '样本构成与固定划分':('sample_composition_and_split','Sample composition'),
 '变量缺失与未知类别':('missing_values_and_unknown_levels','Missing and unknown levels'),
 '主分析_概率校准诊断':('main_calibration_diagnostics','Calibration diagnostics'),
 '主模型DeLong检验与Holm校正':('delong_comparisons_holm','DeLong comparisons'),
 '主模型全部变量SHAP重要性':('main_shap_importance','SHAP importance'),
 'SHAP全流程重拟合稳定性':('shap_refitting_stability','SHAP stability'),
 '训练集点二列相关与校正':('training_point_biserial_correlations','Point-biserial correlations'),
 '训练集方差膨胀因子':('training_variance_inflation_factors','Variance inflation factors'),
 'SHAP重拟合逐次名次':('shap_ranks_by_refit','SHAP ranks by refit'),
 '主分析_开发验证表现与参数':('main_development_metrics_and_parameters','Development metrics'),
 '主分析_校准曲线分组与区间':('main_calibration_bins','Calibration bins'),
 '人口学变量描述统计':('demographic_descriptive_statistics','Demographics'),
 '去除自伤_开发验证表现与参数':('exclusion_development_metrics','Exclusion development'),
 '去除自伤_校准曲线分组与区间':('exclusion_calibration_bins','Exclusion calibration bins'),
 '去除自伤_概率校准诊断':('exclusion_calibration_diagnostics','Exclusion calibration'),
 '去除自伤_测试集完整指标':('exclusion_test_metrics','Exclusion test metrics'),
 '去除自伤前后解释排序比较':('shap_ranks_with_and_without_self_injury','Exclusion SHAP ranks'),
 '变量字典与角色':('variable_dictionary_and_roles','Variable roles'),
 '字段异常处理记录':('field_processing_audit','Field processing'),
 '心理维度描述统计':('psychological_dimension_descriptives','Psychological dimensions'),
 '数据读取与清洗记录':('data_loading_and_cleaning_audit','Data loading audit'),
 '训练阳性率常数概率参照':('constant_probability_reference','Constant probability reference'),
 '训练集Spearman相关矩阵':('training_spearman_correlations','Spearman correlations'),
}

NUMERIC = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
CJK = re.compile(r'[\u3400-\u9fff]')

def translate(text):
    """Translate one exported header or categorical cell; reject omissions."""
    if not isinstance(text, str) or not CJK.search(text):
        return text
    if text in TEXT:
        return TEXT[text]
    if text.endswith('（95%区间）'):
        return translate(text[:-7]) + ' (95% CI)'
    if '|' in text:
        return '|'.join(translate(x) for x in text.split('|'))
    raise ValueError('Untranslated source label: ' + repr(text))

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_tables_dir(source):
    """Find complete original-language inputs in either supported local layout."""
    source = Path(source)
    missing = {}
    for name in ('表格', 'tables'):
        folder = source / name
        if folder.is_dir():
            absent = [basename + '.csv' for basename in TABLES
                      if not (folder / (basename + '.csv')).is_file()]
            if not absent:
                return folder
            missing[name] = absent
    if missing:
        raise FileNotFoundError('Source CSV files must retain their original-language basenames; '
                                'missing required tables: ' + repr(missing))
    raise FileNotFoundError('Source results must contain a tables or 表格 directory')

def export_tables(source, output):
    """Export all 28 group-level tables and preserve every numeric cell string."""
    source_folder = source_tables_dir(source)
    folder = Path(output) / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    checks=[]
    workbook=[]
    for zh, (basename, sheet) in TABLES.items():
        source_path=source_folder/(zh+'.csv')
        with source_path.open(encoding='utf-8-sig',newline='') as fh:
            rows=list(csv.reader(fh))
        translated=[[translate(c) for c in row] for row in rows]
        target=folder/(basename+'.csv')
        with target.open('w',encoding='utf-8-sig',newline='') as fh:
            csv.writer(fh).writerows(translated)
        with target.open(encoding='utf-8-sig',newline='') as fh:
            reread=list(csv.reader(fh))
        assert reread==translated
        numeric=0
        for a,b in zip(rows,reread):
            assert len(a)==len(b)
            for av,bv in zip(a,b):
                if NUMERIC.fullmatch(av) or not av:
                    assert av==bv,(source_path,av,bv)
                    numeric+=1
        assert not any(CJK.search(c) for row in reread for c in row)
        checks.append({'source_table':source_path.name,'english_table':target.name,
                       'rows':max(0,len(rows)-1),'columns':len(rows[0]),
                       'numeric_or_empty_cells_unchanged':numeric,'passed':True,
                       'source_sha256':sha256(source_path),'english_sha256':sha256(target)})
        values=[]
        for i,row in enumerate(translated):
            vals=[]
            for c in row:
                if i==0:vals.append(c)
                elif c=='':vals.append(None)
                elif NUMERIC.fullmatch(c):
                    vals.append(int(c) if re.fullmatch(r'[+-]?\d+',c) else float(c))
                else:vals.append(c)
            values.append(vals)
        workbook.append({'name':sheet,'values':values})
    return checks,workbook
