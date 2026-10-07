@echo off
rem Double-click to build DecisionDashboard.exe (works on a fresh computer).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1"
