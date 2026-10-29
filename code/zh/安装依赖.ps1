param([string]$Python路径 = 'python')
$ErrorActionPreference = 'Stop'
$taskTemp = Join-Path $PSScriptRoot '运行缓存\安装临时目录'
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
& $Python路径 -m pip install --no-cache-dir --target (Join-Path $PSScriptRoot '依赖环境') --requirement (Join-Path $PSScriptRoot '依赖版本.txt')
if ($LASTEXITCODE -ne 0) { throw '依赖安装未完成。' }
