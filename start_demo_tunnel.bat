@echo off
rem =====================================================================
rem  PCBA AOI - one-click live demo via Cloudflare quick tunnel
rem  Starts the Streamlit app locally and exposes it at a temporary public
rem  https://*.trycloudflare.com URL. URL lives only while this window runs.
rem =====================================================================
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo   PCBA AOI live demo - Cloudflare quick tunnel
echo  ============================================================
echo.

rem --- 1. Check cloudflared -------------------------------------------------
where cloudflared >nul 2>nul
if errorlevel 1 (
    echo  [ERROR] cloudflared is not installed.
    echo.
    echo  Install it once, then re-run this script:
    echo     winget install cloudflare.cloudflared
    echo  or download the MSI from:
    echo     https://github.com/cloudflare/cloudflared/releases
    echo     ^(cloudflared-windows-amd64.msi^)
    echo.
    pause
    exit /b 1
)

rem --- 2. Locate the demo venv ----------------------------------------------
set "VENV_PY=.venv-demo\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo  [ERROR] .venv-demo not found at %CD%\.venv-demo
    echo  Run this script from the pcba-aoi project folder.
    echo.
    pause
    exit /b 1
)

rem --- 3. Make sure streamlit is installed in the venv ----------------------
"%VENV_PY%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo  Installing Streamlit into .venv-demo ^(one time^)...
    "%VENV_PY%" -m pip install streamlit
    if errorlevel 1 (
        echo  [ERROR] pip install streamlit failed. Check your network.
        pause
        exit /b 1
    )
)

rem --- 4. Start the app on the demo config ----------------------------------
set "AOI_CONFIG=configs/pipeline.demo.yaml"
echo  Starting the AOI app on http://localhost:8501 ...
start "PCBA AOI app" "%VENV_PY%" -m streamlit run src/app.py --server.headless true --server.port 8501

rem --- 5. Start the tunnel ---------------------------------------------------
echo.
echo  Starting Cloudflare tunnel...
echo  ------------------------------------------------------------
echo   In a few seconds a PUBLIC URL appears below, like:
echo      https://something-random.trycloudflare.com
echo   Share THAT link in your meeting. Anyone can open it while
echo   this window stays open. Close this window to end the demo.
echo  ------------------------------------------------------------
echo.
cloudflared tunnel --url http://localhost:8501

echo.
echo  Tunnel closed - the public URL is no longer reachable.
pause
