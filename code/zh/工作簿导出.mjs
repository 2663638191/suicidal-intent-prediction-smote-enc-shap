// 用公开Artifact Tool API将本次组水平统计表写为可阅读的工作簿。
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const resultDir = process.argv[2];
if (!resultDir) throw new Error('请提供结果目录');
const backend = process.env.WORKBOOK_EXPORT_BACKEND || 'auto';
if (!['auto', 'artifact', 'openpyxl'].includes(backend))
  throw new Error('WORKBOOK_EXPORT_BACKEND must be auto, artifact or openpyxl');
const moduleRoot = process.env.ARTIFACT_NODE_MODULES || process.env.CODEX_ARTIFACT_NODE_MODULES;
const require = moduleRoot ? createRequire(path.join(path.resolve(moduleRoot), 'package.json')) : createRequire(import.meta.url);
let artifact;
if (backend !== 'openpyxl') {
  try { artifact = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href); }
  catch (error) {
    if (backend === 'artifact' || !['MODULE_NOT_FOUND', 'ERR_MODULE_NOT_FOUND'].includes(error.code)) throw error;
  }
}
if (!artifact) {
  const python = process.env.ANALYSIS_PYTHON || process.env.PYTHON || 'python';
  execFileSync(python, [fileURLToPath(new URL('../en/export_workbook.py', import.meta.url)), resultDir, '--input', path.join(resultDir, '复现记录', '工作簿表格输入.json'), '--filename', '论文结果汇总.xlsx', '--record-dir', '复现记录'], {stdio:'inherit'});
  process.exit(0);
}
const { Workbook, SpreadsheetFile } = artifact;
const payload = JSON.parse(await fs.readFile(path.join(resultDir, '复现记录', '工作簿表格输入.json'), 'utf8'));
const wb = Workbook.create();
const previews = path.join(resultDir, '复现记录', '工作簿版面核对');
await fs.mkdir(previews, { recursive: true });
const summaries = [];
for (const item of payload) {
  const sheet = wb.worksheets.add(item.name);
  sheet.showGridLines = false;
  const nr = item.values.length, nc = item.values[0].length;
  const body = sheet.getRangeByIndexes(0, 0, nr, nc);
  body.values = item.values;
  body.format.font = { name: 'Microsoft YaHei', size: 10 };
  body.format.verticalAlignment = 'center';
  body.format.rowHeight = 26;
  body.format.columnWidth = 22;
  body.format.wrapText = true;
  const header = sheet.getRangeByIndexes(0, 0, 1, nc);
  header.format.fill = '#17324D';
  header.format.font = { name: 'Microsoft YaHei', size: 10, bold: true, color: '#FFFFFF' };
  header.format.rowHeight = 44;
  if (nr > 2) sheet.freezePanes.freezeRows(1);
  for (let j = 0; j < nc; j++) {
    const label = String(item.values[0][j]);
    const range = sheet.getRangeByIndexes(1, j, Math.max(1, nr - 1), 1);
    if (item.values.slice(1).some(row => typeof row[j] === 'number')) {
      range.setNumberFormat(/P值/.test(label) ? '0.00E+00' : /人数|记录数|阳性数|阴性数|次数|TN|FP|FN|TP|重复数|源行号|缺失.*数|未知数|^分母$|^名次$/.test(label) ? '#,##0' : '0.0000');
      range.format.horizontalAlignment = 'right';
    }
    if (label.length > 16) sheet.getRangeByIndexes(0, j, nr, 1).format.columnWidth = 30;
    if (['样本', '处理方式'].includes(label)) range.format.horizontalAlignment = 'center';
  }
  for (let row = 2; row < nr; row += 2) sheet.getRangeByIndexes(row, 0, 1, nc).format.fill = '#F2F5F8';
  summaries.push({ name: item.name, rows: nr - 1, columns: nc });
}
wb.recalculate();
const check = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!', options: { useRegex: true, maxResults: 20 } });
await fs.writeFile(path.join(resultDir, '复现记录', '工作簿导出检查.json'), JSON.stringify({ sheets: summaries, errors: check.ndjson }, null, 2));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(resultDir, '论文结果汇总.xlsx'));
for (const [i, item] of payload.entries()) {
  const nr = Math.min(item.values.length, 9), nc = Math.min(item.values[0].length, 7);
  let end = '', q = nc;
  while (q) { q--; end = String.fromCharCode(65 + q % 26) + end; q = Math.floor(q / 26); }
  const preview = await wb.render({ sheetName: item.name, range: `A1:${end}${nr}`, scale: 1.25, format: 'png' });
  await fs.writeFile(path.join(previews, `${String(i + 1).padStart(2, '0')}_${item.name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(`已汇总${payload.length}张组水平结果表并生成各表版面核对图。`);
process.exit(0);
