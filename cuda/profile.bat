@echo off
nvprof --analysis-metrics -o profile\analysis.nvprof -f build\matrixMul.exe -wA=1024 -hA=1024 -wB=1024 -hB=1024
