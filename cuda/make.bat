@echo off

set VCVARSPATH="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set SAMPLES_INC="C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\common\inc" 

REM if cl command is not found, run vcvarsall
where cl >nul 2>nul
if %ERRORLEVEL% neq 0 call %VCVARSPATH%

mkdir build >nul 2>nul

REM Debug build
REM nvcc -ewp -g -G -I%SAMPLES_INC% -o "matrixMul.exe" matrixMul.cu

REM Release build
nvcc -pg -I%SAMPLES_INC% -o "matrixMul.exe" matrixMul.cu

move *.exe build >nul 2>nul
move *.exp build >nul 2>nul
move *.lib build >nul 2>nul
move *.pdb build >nul 2>nul
