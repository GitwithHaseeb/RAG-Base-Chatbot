#!/usr/bin/env bash
set -e

if [ ! -f ".env" ]; then
  echo "Missing .env file. Copy .env.example to .env and add keys."
  exit 1
fi

if [ -f "requirements.txt" ]; then
  python -m pip install -r requirements.txt
fi

streamlit run src/app.py
