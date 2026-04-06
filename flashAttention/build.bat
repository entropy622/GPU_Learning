@echo off
setlocal
nvcc "%~dp0main.cpp" "%~dp0attention_utils.cpp" "%~dp0cpu_cross_attention.cpp" "%~dp0cuda_cross_attention.cu" -o "%~dp0crossAttention.exe" -std=c++17
