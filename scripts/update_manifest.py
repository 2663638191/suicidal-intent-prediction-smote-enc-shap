"""Refresh the inventory of public source files and input definitions."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'docs/repository_manifest.json'
FORBIDDEN = {'.joblib','.pkl','.pickle','.npz','.npy','.doc','.docx','.xlsx','.xls',
             '.pdf','.png','.jpg','.jpeg','.svg','.zip','.tar','.pyc'}
PRIVATE_PARTS = {'results','private','private_records_do_not_publish',
                 '私密记录_禁止公开','依赖环境','运行缓存','__pycache__'}


def main():
    command = ['git','-c','safe.directory=' + ROOT.as_posix(),
               'ls-files','--cached','--others','--exclude-standard','-z']
    data = subprocess.check_output(command,cwd=ROOT).decode('utf-8')
    files = []
    for name in sorted(set(filter(None,data.split('\0')))):
        path = (ROOT/name).resolve()
        if name==MANIFEST or not path.is_file(): continue
        if not path.is_relative_to(ROOT): raise ValueError('Path outside repository')
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN or set(relative.parts) & PRIVATE_PARTS:
            raise ValueError('File outside the source distribution: ' + name)
        if path.suffix.lower()=='.csv' and relative.parts[:2]!=('data','schema'):
            raise ValueError('Only input definitions may be published as CSV: ' + name)
        content = path.read_bytes()
        files.append({'path':relative.as_posix(),'bytes':len(content),
                      'sha256':hashlib.sha256(content).hexdigest()})
    output = {'snapshot':'AJP_source_20261029','hash':'SHA-256',
              'scope':'Source code and input definitions; excludes this inventory itself',
              'files':files}
    (ROOT/MANIFEST).write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Indexed {len(files)} public source and metadata files.')


if __name__=='__main__':
    main()
