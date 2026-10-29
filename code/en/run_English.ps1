param(
    [string]$SourceResults = '',
    [string]$Output = '',
    [string]$DependencyDirectory = '',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $SourceResults) { $SourceResults = Join-Path $taskRepo 'data/private/frozen_results' }
if (-not $Output) { $Output = Join-Path $taskRepo 'results/generated/en' }
$env:PYTHONIOENCODING = 'utf-8'
$arguments = @('-X', 'utf8', (Join-Path $PSScriptRoot 'run_analysis.py'), '--source-results', $SourceResults, '--output', $Output)
if ($DependencyDirectory) { $arguments += @('--dependency-dir', $DependencyDirectory) }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'English export failed; inspect the error before continuing.' }
& $Python -X utf8 (Join-Path $PSScriptRoot 'validate_english_results.py') --source-results $SourceResults --output $Output
if ($LASTEXITCODE -ne 0) { throw 'English output validation failed.' }
