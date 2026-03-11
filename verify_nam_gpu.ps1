$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'activate_nam_gpu_env.ps1')

$pythonExe = 'C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe'
& $pythonExe -c "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device_count=' + str(torch.cuda.device_count())); print('mps_available=' + str(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))"
