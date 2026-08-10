@echo off
REM ===========================================================================
REM 一键本地打包（Windows）
REM   1) 安装 pyinstaller
REM   2) 按 build.spec 构建 onedir 产物到 dist\smart_image_editor\
REM ===========================================================================
setlocal
cd /d "%~dp0"

pip install pyinstaller
IF ERRORLEVEL 1 ( echo [错误] pyinstaller 安装失败 & pause & exit /b 1 )

pyinstaller build.spec --noconfirm --clean
IF ERRORLEVEL 1 ( echo [错误] 构建失败 & pause & exit /b 1 )

echo.
echo [完成] 产物位于: %CD%\dist\smart_image_editor\
echo 首次运行会自动下载模型权重（约 400MB）到 %%APPDATA%%\smart_image_editor\models
pause
