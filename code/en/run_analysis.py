"""Run the complete analysis or export the existing frozen results in English.

The default is a lossless export and figure redraw. Model development only runs
when --fit is supplied explicitly with the original authorised input files.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import shutil
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def configure_dependencies(directory):
    if directory:
        sys.path.insert(0,str(Path(directory).resolve()))
        os.environ['PYTHONPATH']=str(Path(directory).resolve())+os.pathsep+os.environ.get('PYTHONPATH','')
    sys.path.insert(0,str(HERE/'engine'))
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')

def fit_original_analysis(data_dir,output,workers):
    """Run the included full workflow with its original cache fingerprint checks.

    Only the final presentation calls are replaced. All development, validation,
    bootstrap, attribution and sensitivity computations run through core.main.
    Original-language intermediate artefacts remain inside the private cache.
    """
    import contextlib
    import 完整实验 as core
    import 图表输出 as presentation
    import pandas as pd
    import joblib
    from 算法与统计 import 指标名称
    out=Path(output)/'private_records_do_not_publish'/'analysis_cache'
    out.mkdir(parents=True,exist_ok=True)
    original_argv=sys.argv[:]
    old_figures=presentation.create_all_figures
    old_report=presentation.write_report
    presentation.create_all_figures=lambda *a,**kw:None
    presentation.write_report=lambda *a,**kw:'Full analytical computations completed in the private cache.'
    try:
        sys.argv=[str(HERE/'engine'/'完整实验.py'),'--数据目录',str(data_dir),'--结果目录',str(out),
                  '--并行数',str(workers),'--重采样次数','2000','--稳定性次数','100']
        with (out/'engine_execution.log').open('a',encoding='utf-8') as log_file:
            with contextlib.redirect_stdout(log_file),contextlib.redirect_stderr(log_file):
                core.main()
    finally:
        sys.argv=original_argv
        presentation.create_all_figures=old_figures
        presentation.write_report=old_report
    frozen_args=joblib.load(out/'私密记录_禁止公开'/'绘图固定输入.joblib')
    table,ci=frozen_args[9],frozen_args[10]
    rows=[]
    for _,row in table.iterrows():
        item={'模型':row['模型'],'采样策略':row['策略']}
        for name in ['AUC','AP','Recall','Specificity','PPV','NPV','F1','Brier','Review1000','Miss1000']:
            a=ci[(ci['键']==row['键'])&(ci['指标']==指标名称[name])].iloc[0]
            digits=1 if name.endswith('1000') else 3
            item[指标名称[name]+'（95%区间）']=f"{row[name]:.{digits}f} ({a['95%下限']:.{digits}f}, {a['95%上限']:.{digits}f})"
        rows.append(item)
    pd.DataFrame(rows).to_csv(out/'表格'/'论文主结果表_估计值与区间.csv',index=False,encoding='utf-8-sig')
    return out

def write_captions(output,details):
    text='''# Figure captions and display definitions

All numerical inputs are the frozen analytical outputs. The redraw changes labels and layout only. PNG files are exported at 600 dpi; PDF files retain embedded TrueType fonts and vector text/lines. Model abbreviations are LR, logistic regression; DT, decision tree; RF, random forest; ET, extremely randomised trees; XGB, XGBoost; and LGBM, LightGBM.

## Manuscript figures

Figure 1. Receiver operating characteristic curves for six models under (a) no resampling and (b) SMOTE-ENC. Each curve uses frozen test-set probabilities. AUC denotes ROC-AUC.

Figure 2. Precision–recall curves for six models under (a) no resampling and (b) SMOTE-ENC. AP is average precision, calculated as recall-increment-weighted precision. The horizontal dashed line is the test-set system-positive prevalence.

Figure 3. Calibration of the selected model family before and after sigmoid calibration under each resampling strategy. The development validation set supplied the fitted calibration mapping. Points show mean predicted probabilities and observed system-positive proportions in up to ten quantile groups; error bars are Wilson 95% confidence intervals for the observed proportions. Brier scores describe overall probability error.

Figure 4. SHAP beeswarm plot for the ten leading original variables in the selected explanation model. Horizontal position shows contribution to calibrated log-odds. Colour encodes feature value from low to high; nominal categories are grey. Vertical stacking depicts density. The upper display limit is +3; Points beyond this limit are omitted from the view; the omitted-point count is recorded in the run metadata, and their unchanged values remain in the local analytical output.

Figure 5. SHAP waterfall plots for two randomly selected anonymised test records: (a) a predicted-positive record and (b) a predicted-negative record. E[f(x)] is the expected output for the training background sample; f(x) is the record-specific output, both on the calibrated log-odds scale. Red and blue arrows indicate positive and negative contributions. Feature labels show input dimension scores. The remaining contributions are summed as Other features. The existing cases are reused, without selecting for agreement with the system label.

Figure 6. Stability of the ten leading original variables across 100 full-pipeline refits with selected hyperparameters fixed. Bars show top-five and top-ten inclusion frequencies; the right column gives median rank. Training and development records were resampled separately within their roles, while the background and explanation samples were fixed.

## Additional analysis figures

Analysis 01 shows data loading, the fixed training/development/test roles and the analytical workflow. Analysis 02 shows training-set Spearman correlations. Analyses 03 and 06 compare sensitivity and specificity across the twelve model/strategy combinations, with paired outcome-stratified bootstrap percentile 95% confidence intervals. Analyses 04 and 05 show ROC and precision–recall curves. Analysis 07 is the calibration figure; Analysis 08 shows confusion matrices at each frozen threshold. Analyses 09, 10 and 11 show the SHAP beeswarm, individual waterfall plots and refitting stability.

Individual prediction matrices, SHAP matrices, case indices, fitted models and original input data are not included in the public aggregate tables.
'''
    (output/'figure_captions.md').write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description='English outputs for the frozen interpretable screening analysis')
    parser.add_argument('--source-results',type=Path,default=REPO/'data'/'private'/'frozen_results',
                        help='Frozen source tables plus the authorised private plotting cache')
    parser.add_argument('--output',type=Path,default=REPO/'results'/'generated'/'en')
    parser.add_argument('--dependency-dir',default=None)
    parser.add_argument('--fit',action='store_true',help='Run full model development; requires authorised original data')
    parser.add_argument('--data-dir',default=None)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--skip-workbook',action='store_true')
    args=parser.parse_args()
    source=Path(args.source_results).resolve();output=Path(args.output).resolve()
    if source==output or source in output.parents:
        raise ValueError('Use an independent English output directory outside the source results')
    if args.fit and (output/'tables').exists() and any((output/'tables').glob('*.csv')) and not (output/'private_records_do_not_publish'/'analysis_cache'/'复现记录'/'研究配置.json').exists():
        raise ValueError('A full fit requires a fresh output directory; do not overwrite an existing frozen English export')
    if args.fit:
        if not args.data_dir:parser.error('--fit requires --data-dir')
        data_dir=Path(args.data_dir).resolve()
        missing=[f'{year}.xlsx' for year in range(2021,2026) if not (data_dir/f'{year}.xlsx').is_file()]
        if missing:parser.error('Missing authorised input workbooks: '+', '.join(missing))
    elif not (source/'私密记录_禁止公开'/'绘图固定输入.joblib').is_file():
        parser.error('Figure redraw requires the authorised private plotting cache '
                     '(私密记录_禁止公开/绘图固定输入.joblib) under --source-results. '
                     'Aggregate tables alone cannot regenerate individual explanations. '
                     'Use --fit --data-dir with authorised original workbooks to run a new analysis, '
                     'or validate existing local outputs with validate_english_results.py.')
    configure_dependencies(args.dependency_dir)
    for name in ['tables','figures/manuscript','reproducibility']:(output/name).mkdir(parents=True,exist_ok=True)
    if args.fit:
        source=fit_original_analysis(data_dir,output,args.workers)
    from localisation import export_tables,sha256
    checks,workbook=export_tables(source,output)
    (output/'reproducibility'/'workbook_input.json').write_text(json.dumps(workbook,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    import joblib
    from english_figures import create_figures
    cache=source/'私密记录_禁止公开'/'绘图固定输入.joblib'
    frozen_args=joblib.load(cache)
    text_audit,details=create_figures(source,output,frozen_args)
    write_captions(output,details)
    if not args.skip_workbook:
        backend=os.environ.get('WORKBOOK_EXPORT_BACKEND','auto')
        if backend not in {'auto','artifact','openpyxl'}:
            raise ValueError('WORKBOOK_EXPORT_BACKEND must be auto, artifact or openpyxl')
        node=os.environ.get('CODEX_ARTIFACT_NODE') or shutil.which('node')
        if backend=='artifact' and not node:
            raise RuntimeError('The artifact workbook backend requires Node.js')
        if node and backend!='openpyxl':
            subprocess.run([str(node),str(HERE/'export_workbook.mjs'),str(output)],check=True,
                           env=dict(os.environ,ANALYSIS_PYTHON=sys.executable))
        else:
            subprocess.run([sys.executable,str(HERE/'export_workbook.py'),str(output)],check=True)
    report={
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'mode':'full fit and English export' if args.fit else 'frozen-result English export and figure redraw',
        'model_training_rerun':args.fit,'bootstrap_rerun':args.fit,'frozen_plot_cache_sha256':sha256(cache),
        'table_count':len(checks),'table_checks':checks,'numerical_cells_preserved':sum(x['numeric_or_empty_cells_unchanged'] for x in checks),
        'figures':details,'figure_text_contains_cjk':False,'source_results':str(source),'output':str(output),
        'model_input_schema':'Source Chinese fields retained internally; display labels and exported tables use British English',
        'individual_raw_records_copied_to_public_tables':False,
    }
    (output/'reproducibility'/'localisation_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'reproducibility'/'figure_text_audit.json').write_text(json.dumps(text_audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'README.md').write_text("""# Local English analysis outputs

These files were generated from the authorised inputs or cache supplied to the command. The tables folder contains localised aggregate CSV files; figures are written under figures, and an optional editable workbook contains the exported tables.

See figure_captions.md for figure definitions and reproducibility/localisation_validation.json for the checks from this run. Per-record data, fitted objects and private caches remain in their separate local directories. Generated outputs are excluded from the public repository.
""",encoding='utf-8')
    print(json.dumps({'tables':len(checks),'numeric_or_empty_cells_preserved':report['numerical_cells_preserved'],
                     'manuscript_figures':details['manuscript_files'],'training_rerun':args.fit},ensure_ascii=False))

if __name__=='__main__':main()
