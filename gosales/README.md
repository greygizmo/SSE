Quick commands (Windows/PowerShell)

```powershell
$env:PYTHONPATH = "$PWD"; python -m gosales.etl.build_star --config gosales/config.yaml --rebuild
$env:PYTHONPATH = "$PWD"; python gosales/pipeline/score_all.py
```


