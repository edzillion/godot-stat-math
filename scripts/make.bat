@ECHO OFF

REM Command file for Sphinx documentation
REM Note: This script should be run from the project root directory

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=docs
set BUILDDIR=docs\_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

if "%1" == "clean" goto clean

REM Clean build directory and Sphinx cache before building
if "%1" == "html" (
	echo Cleaning build directory and Sphinx cache...
	if exist "%BUILDDIR%" (
		rmdir /s /q "%BUILDDIR%"
		echo Build directory cleared.
	)
	if exist "%SOURCEDIR%\.doctrees" (
		rmdir /s /q "%SOURCEDIR%\.doctrees"
		echo Sphinx doctrees cache cleared.
	)
	echo Building documentation...
)

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
echo Cleaning build directory and Sphinx cache...
if exist "%BUILDDIR%" (
	rmdir /s /q "%BUILDDIR%"
	echo Build directory cleared.
)
if exist "%SOURCEDIR%\.doctrees" (
	rmdir /s /q "%SOURCEDIR%\.doctrees"
	echo Sphinx doctrees cache cleared.
)
echo Clean complete.
goto end

:help
echo Please use `make ^<target^>` where ^<target^> is one of
echo   html       to make standalone HTML files (auto-cleans first)
echo   clean      to clean build directory and Sphinx cache
echo   help       to show this help message
echo.
echo Standard Sphinx targets are also available:
%SPHINXBUILD% -b help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
