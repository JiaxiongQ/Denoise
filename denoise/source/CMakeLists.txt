SET(SOURCE_FILES
DirectWarpDenoiser.cpp
gridTracker.cpp
Fast_klt.cpp
Mesh.cpp
MeshFlow.cpp
MotionDenoiser.cpp
VideoIO.cpp
)
#set(OpenCV_DIR "/usr/local/opencv3/share/OpenCV")
find_package(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
		    ${OpenCV_INCLUDE_DIRS}
		    ${PROJECT_SOURCE_DIR}/include
		   )
	      
add_library(association ${SOURCE_FILES})
target_link_libraries(
association 
${OpenCV_LIBS}
)