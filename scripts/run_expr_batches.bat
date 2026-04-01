@echo off
setlocal EnableExtensions

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_expr_batches.ps1" %*
exit /b %ERRORLEVEL%
