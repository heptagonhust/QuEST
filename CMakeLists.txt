# author: Ania Brown
# author: Jacob Wilkins
# author: Balint Koczor (Windows compatibility)

# CMake initialisation.
cmake_minimum_required(VERSION 3.1)

# Project name
project(QuEST_Project)

# -----------------------------------------------------------------------------
# ----- USER OPTIONS ----------------------------------------------------------
# -----------------------------------------------------------------------------

set(WL1_SOURCE "examples/wl1.c"  CACHE STRING "workload source 1")
set(WL2_SOURCE "examples/wl2.c" CACHE STRING "workload source 2")
set(WL3_SOURCE "examples/wl3.c" CACHE STRING "workload source 3")
set(WL4_SOURCE "examples/wl4.c" CACHE STRING "workload source 4")
set(WL5_SOURCE "examples/wl5.c" CACHE STRING "workload source 5")

set(WL1_EXE "wl1"  CACHE STRING "workload 1 exe")
set(WL2_EXE "wl2" CACHE STRING "workload 2 exe")
set(WL3_EXE "wl3" CACHE STRING "workload 3 exe")
set(WL4_EXE "wl4" CACHE STRING "workload 4 exe")
set(WL5_EXE "wl5" CACHE STRING "workload 5 exe")

add_compile_options(-mavx)
add_compile_options(-mvzeroupper)
add_compile_options(-std=gnu99)

option(TESTING "Enable test suite functionality -- requires test suite available" ON)
set(QuEST_UTIL_DIR_NAME "utilities" CACHE STRING
  "Name of the directory containing the QuEST utilities.")

# -----------------------------------------------------------------------------
# ----- QuEST LIBRARY ---------------------------------------------------------
# -----------------------------------------------------------------------------

# Build the QuEST library if the path to libQuEST.so is not specified
if (NOT DEFINED ${QuEST_LIB_PATH})
    # Build libQuEST.so
    set(QuEST_DIR "QuEST" CACHE STRING
        "Name of the directory containing the QuEST library sources. It must be located in the same directory as the root CMakeLists.txt")
    add_subdirectory(${QuEST_DIR})
    set(QuEST_LIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${QuEST_DIR}")
    set(QuEST_LIB_EXACT "${QuEST_LIB_PATH}/libQuEST.so")
endif()


# -----------------------------------------------------------------------------
# ----- USER EXECUTABLE -------------------------------------------------------
# -----------------------------------------------------------------------------

# Create user executable
add_executable(${WL1_EXE} ${WL1_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${WL1_EXE} QuEST)
else ()
    target_link_libraries(${WL1_EXE} QuEST m)
endif()

# Create user executable
add_executable(${WL2_EXE} ${WL2_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${WL2_EXE} QuEST)
else ()
    target_link_libraries(${WL2_EXE} QuEST m)
endif()

add_executable(${WL3_EXE} ${WL3_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${WL3_EXE} QuEST)
else ()
    target_link_libraries(${WL3_EXE} QuEST m)
endif()

add_executable(${WL4_EXE} ${WL4_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${WL4_EXE} QuEST)
else ()
    target_link_libraries(${WL4_EXE} QuEST m)
endif()

add_executable(${WL5_EXE} ${WL5_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${WL5_EXE} QuEST)
else ()
    target_link_libraries(${WL5_EXE} QuEST m)
endif()

# -----------------------------------------------------------------------------
# ----- UTILS -----------------------------------------------------------------
# -----------------------------------------------------------------------------

set(QuEST_UTIL_DIR "${QuEST_Project_SOURCE_DIR}/${QuEST_UTIL_DIR_NAME}")
# Set dummy UTIL_ROOT location -- Temporary while transitioning to External Project
set(UTIL_ROOT ${QuEST_UTIL_DIR})
# Dummy targets for QuESTPy & QuESTTest due to later inclusion of Utils
add_custom_target( QuESTTest DEPENDS QuEST)
add_custom_target( QuESTPy DEPENDS QuEST)

# -----------------------------------------------------------------------------
# ----- TESTS -----------------------------------------------------------------
# -----------------------------------------------------------------------------

if (${TESTING})
  set(Python_ADDITIONAL_VERSIONS 3.4;3.5;3.6;3.7;3.8)
  find_package(PythonInterp)
  if ((NOT PYTHONINTERP_FOUND) OR (PYTHON_VERSION_MAJOR LESS 3)  OR (PYTHON_VERSION_MINOR LESS 4))
    set(TESTING OFF CACHE BOOL  "Enable test suite functionality -- requires test suite available" FORCE)
    message(WARNING "Usable Python 3.4+ not found on system -- Disabling testing")
  endif()
endif()

message(STATUS "Testing is ${TESTING}")

if (${TESTING})
  enable_testing()
  add_subdirectory(${QuEST_UTIL_DIR})
endif()

# -----------------------------------------------------------------------------
# ----- QuEST Python Interface ------------------------------------------------
# -----------------------------------------------------------------------------

set(QuESTPy_PATH "${QuEST_UTIL_DIR}/QuESTPy" CACHE STRING "Location of QuESTPy")


add_custom_target( update_QuEST_lib_loc
  # Set QuESTPy default dir to this build
  DEPENDS QuESTPy
  COMMAND ${CMAKE_COMMAND} -DQuEST_ROOT=${PROJECT_SOURCE_DIR} -DQuESTPy_PATH=${QuESTPy_PATH} -DQuEST_LIB_EXACT=${QuEST_LIB_EXACT} -DQuEST_LIB_PATH=${QuEST_LIB_PATH} -P ${PROJECT_SOURCE_DIR}/cmake/QuESTPyLib.cmake
  )
