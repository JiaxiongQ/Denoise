# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xxd/Desktop/qjxtemp/denoise

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xxd/Desktop/qjxtemp/denoise/build

# Include any dependencies generated for this target.
include CMakeFiles/VideoDenoiser.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VideoDenoiser.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VideoDenoiser.dir/flags.make

CMakeFiles/VideoDenoiser.dir/main.cpp.o: CMakeFiles/VideoDenoiser.dir/flags.make
CMakeFiles/VideoDenoiser.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xxd/Desktop/qjxtemp/denoise/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VideoDenoiser.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VideoDenoiser.dir/main.cpp.o -c /home/xxd/Desktop/qjxtemp/denoise/main.cpp

CMakeFiles/VideoDenoiser.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VideoDenoiser.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xxd/Desktop/qjxtemp/denoise/main.cpp > CMakeFiles/VideoDenoiser.dir/main.cpp.i

CMakeFiles/VideoDenoiser.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VideoDenoiser.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xxd/Desktop/qjxtemp/denoise/main.cpp -o CMakeFiles/VideoDenoiser.dir/main.cpp.s

CMakeFiles/VideoDenoiser.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/VideoDenoiser.dir/main.cpp.o.requires

CMakeFiles/VideoDenoiser.dir/main.cpp.o.provides: CMakeFiles/VideoDenoiser.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/VideoDenoiser.dir/build.make CMakeFiles/VideoDenoiser.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/VideoDenoiser.dir/main.cpp.o.provides

CMakeFiles/VideoDenoiser.dir/main.cpp.o.provides.build: CMakeFiles/VideoDenoiser.dir/main.cpp.o


# Object files for target VideoDenoiser
VideoDenoiser_OBJECTS = \
"CMakeFiles/VideoDenoiser.dir/main.cpp.o"

# External object files for target VideoDenoiser
VideoDenoiser_EXTERNAL_OBJECTS =

VideoDenoiser: CMakeFiles/VideoDenoiser.dir/main.cpp.o
VideoDenoiser: CMakeFiles/VideoDenoiser.dir/build.make
VideoDenoiser: source/libassociation.a
VideoDenoiser: /usr/local/opencv3/lib/libopencv_objdetect.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_stitching.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_ml.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_dnn.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_superres.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_shape.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_videostab.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_calib3d.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_photo.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_features2d.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_video.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_flann.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_highgui.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_videoio.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_imgcodecs.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_imgproc.so.3.4.1
VideoDenoiser: /usr/local/opencv3/lib/libopencv_core.so.3.4.1
VideoDenoiser: CMakeFiles/VideoDenoiser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xxd/Desktop/qjxtemp/denoise/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VideoDenoiser"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VideoDenoiser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VideoDenoiser.dir/build: VideoDenoiser

.PHONY : CMakeFiles/VideoDenoiser.dir/build

CMakeFiles/VideoDenoiser.dir/requires: CMakeFiles/VideoDenoiser.dir/main.cpp.o.requires

.PHONY : CMakeFiles/VideoDenoiser.dir/requires

CMakeFiles/VideoDenoiser.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VideoDenoiser.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VideoDenoiser.dir/clean

CMakeFiles/VideoDenoiser.dir/depend:
	cd /home/xxd/Desktop/qjxtemp/denoise/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xxd/Desktop/qjxtemp/denoise /home/xxd/Desktop/qjxtemp/denoise /home/xxd/Desktop/qjxtemp/denoise/build /home/xxd/Desktop/qjxtemp/denoise/build /home/xxd/Desktop/qjxtemp/denoise/build/CMakeFiles/VideoDenoiser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VideoDenoiser.dir/depend
