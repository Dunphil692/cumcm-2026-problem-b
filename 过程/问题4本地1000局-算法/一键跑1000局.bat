@echo off
chcp 65001 >nul
cd /d "%~dp0code"
echo 只跑本地 1000 局。不要改代码。
python -m pip install numpy -q
python run_q4_monte_carlo.py --n 1000 --fidelity fast
echo.
echo 跑完后把 code\results 里两个文件发回。
pause
