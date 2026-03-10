$ErrorActionPreference = 'Stop'

$envRoot = 'C:\Users\85014\.conda\envs\nam_gpu_py310'
$pythonExe = Join-Path $envRoot 'python.exe'

if (-not (Test-Path $pythonExe)) {
  throw "GPU environment not found at: $envRoot"
}

$prependPaths = @(
  (Join-Path $envRoot 'Library\bin'),
  (Join-Path $envRoot 'DLLs'),
  (Join-Path $envRoot 'Scripts'),
  $envRoot
)

$env:PATH = ($prependPaths -join ';') + ';' + $env:PATH
$env:CONDA_PREFIX = $envRoot
$env:TF_USE_LEGACY_KERAS = '1'
$env:PYTHONUTF8 = '1'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoParent = Split-Path -Parent $scriptDir
if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
  $env:PYTHONPATH = $repoParent
} else {
  $env:PYTHONPATH = "$repoParent;$env:PYTHONPATH"
}

Write-Host "NAM GPU environment activated."
Write-Host "Python: $pythonExe"
Write-Host "Run with: & `"$pythonExe`" -m neural_additive_models.nam_train <args>"
