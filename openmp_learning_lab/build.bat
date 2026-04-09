@echo off
setlocal

set ROOT=%~dp0
cmake -S "%ROOT%" -B "%ROOT%build" -G "MinGW Makefiles"
if errorlevel 1 exit /b 1

cmake --build "%ROOT%build"
exit /b %errorlevel%
