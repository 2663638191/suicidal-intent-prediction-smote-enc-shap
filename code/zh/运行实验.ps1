param(
    [string]$数据目录 = '',
    [string]$结果目录 = '',
    [int]$并行数 = 4,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $数据目录) { $数据目录 = Join-Path $taskRepo 'data/private/original' }
if (-not $结果目录) { $结果目录 = Join-Path $taskRepo 'results/generated/zh' }
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:LOKY_MAX_CPU_COUNT = [string]$并行数
$taskCache = Join-Path $PSScriptRoot '运行缓存'
$taskTemp = Join-Path $taskCache '临时'
New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
$env:MPLCONFIGDIR = Join-Path $taskCache '绘图库'
$env:NUMBA_CACHE_DIR = Join-Path $taskCache '数值编译'
$taskPython = (Get-Command $Python -ErrorAction Stop).Source
& $taskPython -X utf8 (Join-Path $PSScriptRoot '算法自检.py')
if ($LASTEXITCODE -ne 0) { throw '核心算法自检未通过，终止实验。' }
$taskLogDir = Join-Path $结果目录 '复现记录'
New-Item -ItemType Directory -Path $taskLogDir -Force | Out-Null
& $taskPython -X utf8 -u (Join-Path $PSScriptRoot '完整实验.py') --数据目录 $数据目录 --结果目录 $结果目录 --并行数 $并行数 2>&1 |
    Tee-Object -FilePath (Join-Path $taskLogDir '完整运行日志.txt')
if ($LASTEXITCODE -ne 0) { throw '完整实验未完成，请查看运行日志。' }
& $taskPython -X utf8 (Join-Path $PSScriptRoot '输出复核.py') --结果目录 $结果目录
if ($LASTEXITCODE -ne 0) { throw '输出复核未通过。' }
