$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
if (Test-Path -LiteralPath 'dist/UC_Image_Analysis') { throw 'Build target exists; choose a new build location.' }
python -m PyInstaller --onedir --windowed --name UC_Image_Analysis --paths src --collect-data shap --exclude-module IPython --exclude-module pytest --exclude-module tensorflow src/application/desktop.py
if ($LASTEXITCODE -ne 0) { throw 'Windows build failed.' }
Copy-Item -LiteralPath models -Destination dist/UC_Image_Analysis/models -Recurse
