# tj
TenorJ - a J programming language interface to tensorflow

Ensure you have CMake and a C++20 compatible compiler.

Create the directory structure and files.

### From the j_interpreter directory:
Bash
~~~
mkdir build
cd build
cmake ..
cmake --build .  # Or make, or your generator's build command
~~~
### To run tests (after building):
Bash
~~~
cd build
ctest # Or directly run ./tests/j_interpreter_tests
~~~
### To run the app:
Bash
~~~
cd build
./app/j_repl
~~~