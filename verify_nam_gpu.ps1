$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir 'activate_nam_gpu_env.ps1')

$pythonExe = 'C:\Users\85014\.conda\envs\nam_gpu_py310\python.exe'
& $pythonExe -c "import tensorflow as tf; print('tf=' + tf.__version__); print('built_with_cuda=' + str(tf.test.is_built_with_cuda())); print('gpus=' + str(tf.config.list_physical_devices('GPU')))"
