@echo off
setlocal
set OCL_SET_SVM_SIZE=262144
set PYTHONPATH=%~dp0..
set "MW=%~dp0model_weights"
set "HF_HOME=%MW%\hf_cache"
set "HF_HUB_CACHE=%MW%\hf_cache\hub"
set "HUGGINGFACE_HUB_CACHE=%MW%\hf_cache\hub"
set "TRANSFORMERS_CACHE=%MW%\hf_cache\hub"
set "HF_XET_CACHE=%MW%\hf_cache\xet"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "HF_MODULES_CACHE=%MW%\hf_cache\modules"
set "HF_DATASETS_CACHE=%MW%\hf_datasets"
set "SENTENCE_TRANSFORMERS_HOME=%MW%\sentence_transformers"
set "TORCH_HOME=%MW%\torch_cache"
set "TRITON_CACHE_DIR=%MW%\triton"
set "NUMBA_CACHE_DIR=%MW%\numba"
set "ULTRALYTICS_HOME=%MW%"
set "YOLO_CONFIG_DIR=%MW%"
set "MPLCONFIGDIR=%MW%\matplotlib"
set "XDG_CACHE_HOME=%MW%\cache"
set "XDG_CONFIG_HOME=%MW%\config"
set "XDG_DATA_HOME=%MW%\data"
set "TMPDIR=%MW%\tmp"
set "TEMP=%MW%\tmp"
set "TMP=%MW%\tmp"
set "MIOPEN_USER_DB_PATH=%MW%\miopen"
set "MIOPEN_CUSTOM_CACHE_DIR=%MW%\miopen\cache"
set "MIOPEN_FIND_MODE=2"
set "MIOPEN_FIND_ENFORCE=1"
set "MIOPEN_SEARCH_CUTOFF=1"
set "MIOPEN_DEBUG_DISABLE_FIND_DB=0"
if not defined TORCH_BLAS_PREFER_HIPBLASLT set "TORCH_BLAS_PREFER_HIPBLASLT=1"
if not defined DISABLE_ADDMM_CUDA_LT set "DISABLE_ADDMM_CUDA_LT=1"
if not defined SPECTRA_DETECTOR_FP16 set "SPECTRA_DETECTOR_FP16=1"
set "PYTORCH_TUNABLEOP_CACHE_DIR=%MW%\tunableop_cache"
if not exist "%MW%\tmp" md "%MW%\tmp" 2>nul
if not exist "%MW%\matplotlib" md "%MW%\matplotlib" 2>nul
if not exist "%MW%\cache" md "%MW%\cache" 2>nul
if not exist "%MW%\config" md "%MW%\config" 2>nul
if not exist "%MW%\data" md "%MW%\data" 2>nul
if not exist "%MW%\hf_datasets" md "%MW%\hf_datasets" 2>nul
if not exist "%MW%\sentence_transformers" md "%MW%\sentence_transformers" 2>nul
if not exist "%MW%\triton" md "%MW%\triton" 2>nul
if not exist "%MW%\numba" md "%MW%\numba" 2>nul
if not exist "%MW%\miopen\cache" md "%MW%\miopen\cache" 2>nul
if not exist "%MW%\tunableop_cache" md "%MW%\tunableop_cache" 2>nul
if defined SPECTRA_OMP_THREADS (set "OMP_NUM_THREADS=%SPECTRA_OMP_THREADS%") else (for /f %%a in ('powershell -NoProfile -Command "$c=0; Get-CimInstance Win32_Processor | ForEach-Object { $c += $_.NumberOfCores }; [Math]::Max(1,[Math]::Min(6,$c))"') do set "OMP_NUM_THREADS=%%a")
set "KMP_AFFINITY=granularity=fine,compact,1,0"
set "KMP_BLOCKTIME=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
"%~dp0.venv\Scripts\python.exe" -X pycache_prefix="%MW%\pycache" -W ignore::DeprecationWarning -W ignore::FutureWarning -m uvicorn backend_amd_gpu.main:app --host 127.0.0.1 --port 8000 --log-level error
