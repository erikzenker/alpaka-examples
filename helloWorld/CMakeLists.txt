project(HASEonGPU)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
  "$ENV{MPI_ROOT}" "$ENV{BOOST_ROOT}")


if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.2)
  message(FATAL_ERROR "GCC version must be at least 4.8.2")
endif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.2)


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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnarrowing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Waddress")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Warray-bounds")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wchar-subscripts")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcomment")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmain")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmissing-braces")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wparentheses")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreturn-type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsequence-point")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-overflow=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wtrigraphs")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-function")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-label")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-value")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wvolatile-register-var")
  # new warning in gcc 4.8 (flag ignored in previous version)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
  # ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
  # PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
  
  # Visual Studio (Microsoft)
 elseif(MSVC)
    # This loop sets all linking to static (MT)
    foreach(flag_var
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif()


###############################################################################
# Executables
###############################################################################
set(HelloWorldKernel "helloWorldKernel")
file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(${HelloWorldKernel} ${SRCFILES})
target_link_libraries(${HelloWorldKernel} ${LIBS})
