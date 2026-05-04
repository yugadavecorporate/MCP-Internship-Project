@echo off
title MCP Chatbot Launcher
echo Installing required dependencies...
pip install -r requirements.txt

echo.
echo Starting Streamlit Chatbot...
streamlit run cli_chatbot.py

pause
