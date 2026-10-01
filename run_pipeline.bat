@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 1251 >nul
title FET-PET

rem Double-click: menu. Command line passes arguments straight to run_pipeline.py.
rem run_pipeline.bat --t20 DIR20 --t40 DIR40 --t60 DIR60 --t1 DIRT1 --motion-correct -o OUT
rem run_pipeline.bat --dynamic --dyn1 A --dyn2 B --dyn3 C --static-ref S --t1 T --motion-correct -o OUT

set "PY=%~dp0.venv\Scripts\python.exe"
set "SCRIPT=%~dp0run_pipeline.py"

if not exist "%PY%" goto :no_venv
if not exist "%SCRIPT%" goto :no_script

"%PY%" -c "import dcm2niix" >nul 2>&1
set "HAS_MOD=%ERRORLEVEL%"
where dcm2niix >nul 2>&1
if not "%HAS_MOD%"=="0" if errorlevel 1 (
    echo [предупреждение] dcm2niix не найден ни в .venv, ни в PATH.
    echo.
)

if not "%~1"=="" goto :direct

:menu
echo.
echo FET-PET / MRI
echo   1  Статика    20 / 40 / 60 мин
echo   2  Динамика   3 серии
echo   0  Выход
echo.
set "MODE="
set /p "MODE=Режим [1]: "
if "%MODE%"=="" set "MODE=1"
if "%MODE%"=="0" exit /b 0
if "%MODE%"=="1" goto :static
if "%MODE%"=="2" goto :dynamic
echo Неизвестный режим.
goto :menu

:static
echo.
echo Статика. Путь из проводника можно вставлять вместе с кавычками.
call :ask T20 "Папка 20 мин"
if errorlevel 1 goto :menu
call :ask T40 "Папка 40 мин"
if errorlevel 1 goto :menu
call :ask T60 "Папка 60 мин"
if errorlevel 1 goto :menu
call :ask_opt T1 "Папка T1, Enter — без МРТ"
if errorlevel 1 goto :menu
call :ask_extra static
call :default_out OUT "%T20%"
call :ask_opt OUT "Папка результата [%OUT%]"
if not defined OUT call :default_out OUT "%T20%"

set "ARGS="
call :add --t20 "%T20%"
call :add --t40 "%T40%"
call :add --t60 "%T60%"
call :add -o "%OUT%"
if defined T1 call :add --t1 "%T1%"
if defined EXTRAARGS set ARGS=%ARGS% %EXTRAARGS%
goto :run

:dynamic
echo.
echo Динамика. Три папки с 4D-сериями, по порядку времени.
call :ask DYN1 "Серия 1, ранняя"
if errorlevel 1 goto :menu
call :ask DYN2 "Серия 2"
if errorlevel 1 goto :menu
call :ask DYN3 "Серия 3, поздняя"
if errorlevel 1 goto :menu
call :ask_opt SREF "Статика для веса, роста и дозы. Enter — пропустить"
if errorlevel 1 goto :menu
call :ask_opt T1 "Папка T1, Enter — без МРТ"
if errorlevel 1 goto :menu
call :ask_extra dynamic
call :default_out OUT "%DYN1%"
call :ask_opt OUT "Папка результата [%OUT%]"
if not defined OUT call :default_out OUT "%DYN1%"

set "ARGS="
call :add --dynamic
call :add --dyn1 "%DYN1%"
call :add --dyn2 "%DYN2%"
call :add --dyn3 "%DYN3%"
call :add -o "%OUT%"
if defined SREF call :add --static-ref "%SREF%"
if defined T1 call :add --t1 "%T1%"
if defined EXTRAARGS set ARGS=%ARGS% %EXTRAARGS%
goto :run

:run
echo.
echo "%PY%" -u "%SCRIPT%" %ARGS%
echo.
set "GO="
set /p "GO=Enter — запуск, n — отмена: "
if /i "%GO%"=="n" goto :menu
if /i "%GO%"=="no" goto :menu
if /i "%GO%"=="нет" goto :menu
echo.
"%PY%" -u "%SCRIPT%" %ARGS%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo Готово. Результат: %OUT%
if not "%RC%"=="0" echo Ошибка, код %RC%.
echo.
set "AGAIN="
set /p "AGAIN=Ещё одно исследование? [y/N]: "
if /i "%AGAIN%"=="y" goto :menu
exit /b %RC%

:direct
"%PY%" -u "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Ошибка, код %RC%.
    pause
)
exit /b %RC%

:ask
set "%~1="
set /p "%~1=%~2: "
call :clean_path %~1
if not defined %~1 echo Папка не указана.
if not defined %~1 exit /b 1
call set "CHK=%%%~1%%"
if exist "%CHK%\." exit /b 0
echo Папка не найдена: %CHK%
exit /b 1

:ask_opt
set "%~1="
set /p "%~1=%~2: "
call :strip %~1
if not defined %~1 exit /b 0
if /i "%~1"=="NOFRAME" exit /b 0
if /i "%~1"=="SIGFROM" exit /b 0
call :clean_path %~1
if /i "%~1"=="OUT" exit /b 0
call set "CHK=%%%~1%%"
if exist "%CHK%\." exit /b 0
echo Папка не найдена: %CHK%
exit /b 1

:ask_yn
set "%~1=1"
set "YN="
set /p "YN=%~2 [Y/n]: "
if /i "%YN%"=="n" set "%~1=0"
if /i "%YN%"=="no" set "%~1=0"
if /i "%YN%"=="нет" set "%~1=0"
exit /b 0

:ask_no
set "%~1=0"
set "YN="
set /p "YN=%~2 [y/N]: "
if /i "%YN%"=="y" set "%~1=1"
if /i "%YN%"=="yes" set "%~1=1"
if /i "%YN%"=="д" set "%~1=1"
if /i "%YN%"=="да" set "%~1=1"
exit /b 0

:ask_extra
set "EXTRAARGS="
echo.
call :ask_no SETEXTRA "Задать пороги и прочие параметры?"
if "%SETEXTRA%"=="0" exit /b 0
echo Enter оставляет значение в скобках. Дробные числа — через точку.
call :ask_def SULTH "Порог SUL (--sul-threshold)" "2.0"
call :addx --sul-threshold "%SULTH%"
call :ask_def TBRTH "Порог TBR (--tbr-threshold)" "2.0"
call :addx --tbr-threshold "%TBRTH%"
call :ask_def TBRDELTA "Мин. изменение TBR для тренда (--tbr-delta-threshold)" "0.3"
call :addx --tbr-delta-threshold "%TBRDELTA%"
if /i "%~1"=="static" goto :extra_static_time
goto :extra_common
:extra_static_time
call :ask_def TIMESPAN "Интервал первой и последней точки, мин (--time-span)" "40"
call :addx --time-span "%TIMESPAN%"
call :ask_def TIMEPTS "Временные точки, мин, три числа (--time-points)" "20 40 60"
call :addx --time-points %TIMEPTS%
:extra_common
call :ask_def TRIM "Обрезка хвостов SULmean, проценты (--trim-percent)" "2.5"
call :addx --trim-percent "%TRIM%"
call :ask_def CLUSTER "Мин. кластер, воксели. 0 выключает (--min-cluster-size)" "45"
call :addx --min-cluster-size "%CLUSTER%"
call :ask_def CORDMM "Продлить маску от ствола вниз по спинному мозгу, мм (--cord-mm)" "0"
if not "%CORDMM%"=="0" if not "%CORDMM%"=="0.0" call :addx --cord-mm "%CORDMM%"
call :ask_yn SKULL "Удаление черепа"
if "%SKULL%"=="0" call :addx --no-skull-strip
call :ask_yn SMOOTH "Сглаживание"
if "%SMOOTH%"=="0" call :addx --no-smoothing
if not "%SMOOTH%"=="1" goto :extra_motion
call :ask_def SIGMA "Сигма сглаживания, воксели (--smooth-sigma)" "1.0"
call :addx --smooth-sigma "%SIGMA%"
:extra_motion
call :ask_no MOTION "Коррекция движения (--motion-correct)"
if "%MOTION%"=="1" call :addx --motion-correct
if /i not "%~1"=="dynamic" exit /b 0
call :ask_def NOFRAME "Пропустить первые N кадров (--no-frame)" "0"
call :addx --no-frame "%NOFRAME%"
call :ask_def SIGFROM "Маска и наклон только с этой минуты (--sig-time-from)" "0"
call :addx --sig-time-from "%SIGFROM%"
exit /b 0

:ask_def
set "%~1=%~3"
set /p "%~1=%~2 [%~3]: "
call :strip %~1
if not defined %~1 set "%~1=%~3"
exit /b 0

:default_out
for %%I in ("%~2\..") do set "%~1=%%~fI\output"
exit /b 0

:add
set ARGS=%ARGS% %*
exit /b 0

:addx
set EXTRAARGS=%EXTRAARGS% %*
exit /b 0

:strip
call set "VAL=%%%~1%%"
if not defined VAL exit /b 0
set VAL=%VAL:"=%
set "%~1=%VAL%"
exit /b 0

:clean_path
call :strip %~1
if not defined %~1 exit /b 0
call set "VAL=%%%~1%%"
for %%A in ("%VAL%\.") do set "%~1=%%~fA"
exit /b 0

:no_venv
echo Не найдено окружение .venv
echo Ожидается: %PY%
echo.
echo В папке со скриптом выполните:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install "numpy<2.4" antspyx antspynet nibabel pydicom scipy matplotlib PyQt5 dcm2niix
pause
exit /b 1

:no_script
echo Рядом с bat нет run_pipeline.py
echo Папка: %~dp0
pause
exit /b 1
