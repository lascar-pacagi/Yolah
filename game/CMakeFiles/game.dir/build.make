# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/elucterio/IA/Option/Code/OptionProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/elucterio/IA/Option/Code/OptionProject

# Include any dependencies generated for this target.
include game/CMakeFiles/game.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include game/CMakeFiles/game.dir/compiler_depend.make

# Include the progress variables for this target.
include game/CMakeFiles/game.dir/progress.make

# Include the compile flags for this target's objects.
include game/CMakeFiles/game.dir/flags.make

game/CMakeFiles/game.dir/game.cpp.o: game/CMakeFiles/game.dir/flags.make
game/CMakeFiles/game.dir/game.cpp.o: game/game.cpp
game/CMakeFiles/game.dir/game.cpp.o: game/CMakeFiles/game.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elucterio/IA/Option/Code/OptionProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object game/CMakeFiles/game.dir/game.cpp.o"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT game/CMakeFiles/game.dir/game.cpp.o -MF CMakeFiles/game.dir/game.cpp.o.d -o CMakeFiles/game.dir/game.cpp.o -c /home/elucterio/IA/Option/Code/OptionProject/game/game.cpp

game/CMakeFiles/game.dir/game.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/game.dir/game.cpp.i"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elucterio/IA/Option/Code/OptionProject/game/game.cpp > CMakeFiles/game.dir/game.cpp.i

game/CMakeFiles/game.dir/game.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/game.dir/game.cpp.s"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elucterio/IA/Option/Code/OptionProject/game/game.cpp -o CMakeFiles/game.dir/game.cpp.s

game/CMakeFiles/game.dir/magic.cpp.o: game/CMakeFiles/game.dir/flags.make
game/CMakeFiles/game.dir/magic.cpp.o: game/magic.cpp
game/CMakeFiles/game.dir/magic.cpp.o: game/CMakeFiles/game.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elucterio/IA/Option/Code/OptionProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object game/CMakeFiles/game.dir/magic.cpp.o"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT game/CMakeFiles/game.dir/magic.cpp.o -MF CMakeFiles/game.dir/magic.cpp.o.d -o CMakeFiles/game.dir/magic.cpp.o -c /home/elucterio/IA/Option/Code/OptionProject/game/magic.cpp

game/CMakeFiles/game.dir/magic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/game.dir/magic.cpp.i"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elucterio/IA/Option/Code/OptionProject/game/magic.cpp > CMakeFiles/game.dir/magic.cpp.i

game/CMakeFiles/game.dir/magic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/game.dir/magic.cpp.s"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elucterio/IA/Option/Code/OptionProject/game/magic.cpp -o CMakeFiles/game.dir/magic.cpp.s

game/CMakeFiles/game.dir/move.cpp.o: game/CMakeFiles/game.dir/flags.make
game/CMakeFiles/game.dir/move.cpp.o: game/move.cpp
game/CMakeFiles/game.dir/move.cpp.o: game/CMakeFiles/game.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elucterio/IA/Option/Code/OptionProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object game/CMakeFiles/game.dir/move.cpp.o"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT game/CMakeFiles/game.dir/move.cpp.o -MF CMakeFiles/game.dir/move.cpp.o.d -o CMakeFiles/game.dir/move.cpp.o -c /home/elucterio/IA/Option/Code/OptionProject/game/move.cpp

game/CMakeFiles/game.dir/move.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/game.dir/move.cpp.i"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elucterio/IA/Option/Code/OptionProject/game/move.cpp > CMakeFiles/game.dir/move.cpp.i

game/CMakeFiles/game.dir/move.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/game.dir/move.cpp.s"
	cd /home/elucterio/IA/Option/Code/OptionProject/game && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elucterio/IA/Option/Code/OptionProject/game/move.cpp -o CMakeFiles/game.dir/move.cpp.s

game: game/CMakeFiles/game.dir/game.cpp.o
game: game/CMakeFiles/game.dir/magic.cpp.o
game: game/CMakeFiles/game.dir/move.cpp.o
game: game/CMakeFiles/game.dir/build.make
.PHONY : game

# Rule to build all files generated by this target.
game/CMakeFiles/game.dir/build: game
.PHONY : game/CMakeFiles/game.dir/build

game/CMakeFiles/game.dir/clean:
	cd /home/elucterio/IA/Option/Code/OptionProject/game && $(CMAKE_COMMAND) -P CMakeFiles/game.dir/cmake_clean.cmake
.PHONY : game/CMakeFiles/game.dir/clean

game/CMakeFiles/game.dir/depend:
	cd /home/elucterio/IA/Option/Code/OptionProject && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/elucterio/IA/Option/Code/OptionProject /home/elucterio/IA/Option/Code/OptionProject/game /home/elucterio/IA/Option/Code/OptionProject /home/elucterio/IA/Option/Code/OptionProject/game /home/elucterio/IA/Option/Code/OptionProject/game/CMakeFiles/game.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : game/CMakeFiles/game.dir/depend

