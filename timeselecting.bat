@echo off
set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%"
set "ANACONDAPATH=%ANACONDA_PATH%"

::cd /d "%SCRIPT_DIR%"
%ANACONDAPATH%\python -c "from signal_update_main import update_main; update_main()"

