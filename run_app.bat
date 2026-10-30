@echo off
setlocal enabledelayedexpansion
title KNN Patch Denoiser Studio

echo ========================================================
echo        KNN Patch Denoiser Studio - Launcher
echo ========================================================
echo.

:: 1. Memeriksa keberadaan Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak ditemukan di sistem Anda!
    echo Silakan install Python terlebih dahulu dari https://www.python.org/
    echo Pastikan untuk mencentang opsi "Add Python to PATH" saat instalasi.
    echo.
    pause
    exit /b 1
)

:: 2. Memeriksa dependensi pustaka Python
echo [*] Memeriksa dependensi pustaka Python...
python -c "import streamlit, numpy, PIL, matplotlib, scipy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Beberapa dependensi belum terpasang atau belum lengkap.
    echo [*] Memulai proses instalasi otomatis dari requirements.txt...
    echo.
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Gagal menginstall dependensi melalui pip.
        echo Periksa koneksi internet Anda lalu coba lagi.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Seluruh dependensi berhasil dipasang!
) else (
    echo [OK] Seluruh dependensi sudah terpasang dengan lengkap.
)

echo.
echo ========================================================
echo   Menjalankan Streamlit KNN Denoiser Studio...
echo ========================================================
echo.
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Aplikasi telah dihentikan.
    pause
)
