@echo off
rem Scheduled collect (README: "Collecting quotes on a schedule"). Runs from the repo folder; output appended to logs\collect_task.log.
cd /d "%~dp0.."
echo ===== %DATE% %TIME% ===== >> logs\collect_task.log
python run.py collect >> logs\collect_task.log 2>&1
echo exit code %ERRORLEVEL% >> logs\collect_task.log
