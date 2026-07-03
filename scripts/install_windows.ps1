Write-Host "[LAYER] Windows install helper for lightweight syntax/report checks only."
python --version
python -c "import importlib; names=['numpy','PIL','matplotlib']; [print(f'{name}: '+str(getattr(importlib.import_module(name), '__version__', 'available'))) if importlib.util.find_spec(name) else print(f'{name}: missing') for name in names]"
Write-Host "CUDA InstructPix2Pix scans should be run on the A6000 environment."
