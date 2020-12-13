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

set(RANDORM_SOURCE  "examples/random.c"  CACHE STRING "Source to build with QuEST library")

set(QFT_SOURCE "examples/GHZ_QFT.c" CACHE STRING "QFT")

set(RANDOM_OUTPUT_EXE   "demo"  CACHE STRING "Executable to compile to")
set(QFT_OUTPUT_EXE "qft" CACHE STRING "qft executable")

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

message(STATUS "Compiling ${RANDORM_SOURCE} to executable ${RANDOM_OUTPUT_EXE}")


# Create user executable
add_executable(${RANDOM_OUTPUT_EXE} ${RANDORM_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${RANDOM_OUTPUT_EXE} QuEST)
else ()
    target_link_libraries(${RANDOM_OUTPUT_EXE} QuEST m)
endif()

message(STATUS "Compiling ${QFT_SOURCE} to executable ${QFT_OUTPUT_EXE}")


# Create user executable
add_executable(${QFT_OUTPUT_EXE} ${RANDORM_SOURCE})

# Link libraries to user executable, including QuEST library
if (WIN32)
    target_link_libraries(${QFT_OUTPUT_EXE} QuEST)
else ()
    target_link_libraries(${QFT_OUTPUT_EXE} QuEST m)
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
