# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/geodbpy.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/geodbpy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/geodbpy.dir/flags.make

CMakeFiles/geodbpy.dir/src/PyModule.cpp.o: CMakeFiles/geodbpy.dir/flags.make
CMakeFiles/geodbpy.dir/src/PyModule.cpp.o: ../src/PyModule.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/geodbpy.dir/src/PyModule.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/geodbpy.dir/src/PyModule.cpp.o -c /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/src/PyModule.cpp

CMakeFiles/geodbpy.dir/src/PyModule.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/geodbpy.dir/src/PyModule.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/src/PyModule.cpp > CMakeFiles/geodbpy.dir/src/PyModule.cpp.i

CMakeFiles/geodbpy.dir/src/PyModule.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/geodbpy.dir/src/PyModule.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/src/PyModule.cpp -o CMakeFiles/geodbpy.dir/src/PyModule.cpp.s

# Object files for target geodbpy
geodbpy_OBJECTS = \
"CMakeFiles/geodbpy.dir/src/PyModule.cpp.o"

# External object files for target geodbpy
geodbpy_EXTERNAL_OBJECTS =

geodbpy.cpython-38-darwin.so: CMakeFiles/geodbpy.dir/src/PyModule.cpp.o
geodbpy.cpython-38-darwin.so: CMakeFiles/geodbpy.dir/build.make
geodbpy.cpython-38-darwin.so: libgeodb.a
geodbpy.cpython-38-darwin.so: /Users/joeyg/docs/workspace/libtorch/lib/libtorch.dylib
geodbpy.cpython-38-darwin.so: /Users/joeyg/docs/workspace/libtorch/lib/libtorch_cpu.dylib
geodbpy.cpython-38-darwin.so: /Users/joeyg/docs/workspace/libtorch/lib/libc10.dylib
geodbpy.cpython-38-darwin.so: /Users/joeyg/docs/workspace/libtorch/lib/libc10.dylib
geodbpy.cpython-38-darwin.so: CMakeFiles/geodbpy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module geodbpy.cpython-38-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/geodbpy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/geodbpy.dir/build: geodbpy.cpython-38-darwin.so

.PHONY : CMakeFiles/geodbpy.dir/build

CMakeFiles/geodbpy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/geodbpy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/geodbpy.dir/clean

CMakeFiles/geodbpy.dir/depend:
	cd /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug /Users/joeyg/Dropbox/Documents/workspace/GeoDb/geodbcpp/cmake-build-debug/CMakeFiles/geodbpy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/geodbpy.dir/depend
