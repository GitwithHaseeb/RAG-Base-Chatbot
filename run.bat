@echo off
setlocal

if not exist ".env" (
  echo Missing .env file. Copy .env.example to .env and add keys.
  exit /b 1
)

if exist "requirements.txt" (
  python -m pip install -r requirements.txt
)

streamlit run src/app.py
