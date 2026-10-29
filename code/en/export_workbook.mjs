// Export the English group-level tables using the supported Artifact Tool API.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const output=process.argv[2];
if(!output)throw new Error('Provide the English result directory');
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
  execFileSync(python, [fileURLToPath(new URL('./export_workbook.py', import.meta.url)), output], {stdio:'inherit'});
  process.exit(0);
}
const { Workbook, SpreadsheetFile } = artifact;
const payload=JSON.parse(await fs.readFile(path.join(output,'reproducibility','workbook_input.json'),'utf8'));
const previews=path.join(output,'reproducibility','workbook_previews');
await fs.mkdir(previews,{recursive:true});
const wb=Workbook.create();
const sheets=[];
for(const item of payload){
 const sheet=wb.worksheets.add(item.name);
 sheet.showGridLines=false;
 const nr=item.values.length,nc=item.values[0].length;
 const body=sheet.getRangeByIndexes(0,0,nr,nc);
 body.values=item.values;
 body.format.font={name:'Arial',size:10};
 body.format.verticalAlignment='center';
 body.format.rowHeight=30;
 body.format.columnWidth=24;
 body.format.wrapText=true;
 const header=sheet.getRangeByIndexes(0,0,1,nc);
 header.format.fill='#17324D';
 header.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
 header.format.rowHeight=84;
 header.format.horizontalAlignment='center';
 if(nr>9)sheet.freezePanes.freezeRows(1);
 for(let j=0;j<nc;j++){
  const label=String(item.values[0][j]);
  const range=sheet.getRangeByIndexes(1,j,Math.max(1,nr-1),1);
  const vals=item.values.slice(1).map(r=>r[j]);
  if(vals.some(v=>typeof v==='number')){
   const allInteger=vals.filter(v=>typeof v==='number').every(Number.isInteger);
   range.setNumberFormat(/(?:^|\s)P value$/.test(label)?'0.00E+00':allInteger?'#,##0':'0.0000');
   range.format.horizontalAlignment='right';
  }else{
   const maxText=Math.max(0,...vals.map(v=>typeof v==='string'?v.length:0));
   if(maxText>35)sheet.getRangeByIndexes(0,j,nr,1).format.columnWidth=42;
   if(maxText>90){
    sheet.getRangeByIndexes(0,j,nr,1).format.columnWidth=66;
    for(let i=1;i<nr;i++)if(typeof item.values[i][j]==='string'&&item.values[i][j].length>90)
      sheet.getRangeByIndexes(i,0,1,nc).format.rowHeight=Math.min(160,Math.max(44,Math.ceil(item.values[i][j].length/85)*15));
   }
  }
 }
 for(let i=2;i<nr;i+=2)sheet.getRangeByIndexes(i,0,1,nc).format.fill='#F2F5F8';
 sheets.push({name:item.name,rows:nr-1,columns:nc});
}
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20}});
await fs.writeFile(path.join(output,'reproducibility','workbook_export_checks.json'),JSON.stringify({sheets,error_scan:errors.ndjson},null,2));
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(output,'analysis_results_English.xlsx'));
for(let i=0;i<payload.length;i++){
 const item=payload[i],nr=Math.min(item.values.length,7),nc=Math.min(item.values[0].length,6);
 const end=String.fromCharCode(64+nc);
 const preview=await wb.render({sheetName:item.name,range:`A1:${end}${nr}`,scale:1.1,format:'png'});
 await fs.writeFile(path.join(previews,`${String(i+1).padStart(2,'0')}_${item.name.replace(/[^A-Za-z0-9]+/g,'_')}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const inspectionSidecar=path.join(output,'analysis_results_English.xlsx.inspect.ndjson');
try{await fs.rename(inspectionSidecar,path.join(output,'reproducibility','workbook_inspection.ndjson'));}
catch(error){if(error.code!=='ENOENT')throw error;}
console.log(`Exported ${payload.length} English worksheets and their layout previews.`);
process.exit(0);
