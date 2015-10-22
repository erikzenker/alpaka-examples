project(alpaka-example-helloworld)
cmake_minimum_required(VERSION 3.0.1)

################################################################################
# Find alpaka
################################################################################
SET(ALPAKA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../alpaka" CACHE STRING  "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")

find_package("alpaka" REQUIRED)
include_directories(SYSTEM ${alpaka_INCLUDE_DIRS})
add_definitions(${alpaka_DEFINITIONS}) 
set(LIBS ${LIBS} ${alpaka_LIBRARIES})

################################################################################
# Compiler Flags
################################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

###############################################################################
# Executables
###############################################################################
set(HelloWorldKernel "helloWorldKernel")
file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${HelloWorldKernel} ${SRCFILES})
target_link_libraries(${HelloWorldKernel} ${LIBS})
