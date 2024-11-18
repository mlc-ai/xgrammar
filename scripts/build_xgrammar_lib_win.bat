echo on

del /f config.cmake
echo set(XGRAMMAR_BUILD_PYTHON_BINDINGS ON) >> config.cmake
echo set(XGRAMMAR_BUILD_CXX_TESTS OFF) >> config.cmake

rd /s /q build
mkdir build
cd build

cmake -A x64 -Thost=x64 ^
      -G "Visual Studio 17 2022" ^
      ..

if %errorlevel% neq 0 exit %errorlevel%

cmake --build . --parallel 3 --config Release --target xgrammar_bindings -- /m

if %errorlevel% neq 0 exit %errorlevel%

cd ..
