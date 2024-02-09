
cmake_minimum_required(VERSION 3.25)

# ---------- Languages and options ----------
project(Hubbard LANGUAGES CXX)

option(HUBBARD_TESTING "" ON)
option(HUBBARD_UPDATE_SUBMODULES "" OFF)
option(HUBBARD_USE_CUDA "" ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(HUBBARD_USE_CUDA)
   include(CheckLanguage)
   check_language(CUDA)
   if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
   
      # NOTE: CUDA options are set globally
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      set(CMAKE_CUDA_ARCHITECTURES native)
      set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

      find_package(CUDAToolkit REQUIRED)
   else()
      set(HUBBARD_USE_CUDA OFF CACHE BOOL "" FORCE)
   endif()
endif()
set(use_cuda $<BOOL:${HUBBARD_USE_CUDA}>)

# ---------- Build options ----------
set(CMAKE_COLOR_DIAGNOSTICS ON)
add_library(hubbard_compiler_flags INTERFACE)
add_library(hubbard_linker_flags INTERFACE)

target_compile_features(hubbard_compiler_flags INTERFACE cxx_std_20)
target_compile_definitions(hubbard_compiler_flags INTERFACE "$<$<CONFIG:DEBUG>:HUBBARD_DEBUG>" "$<${use_cuda}:HUBBARD_USE_CUDA>")

function(set_comp_flags config lang flags)
   target_compile_options(hubbard_compiler_flags INTERFACE "$<$<CONFIG:${config}>:$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:${lang}>:${flags}>>>")
endfunction()

function(set_link_flags config lang flags)
   target_link_options(hubbard_linker_flags INTERFACE "$<$<CONFIG:${config}>:$<BUILD_INTERFACE:$<$<LINK_LANGUAGE:${lang}>:${flags}>>>")
endfunction()

# NOTE: CUDA host compiler should match CXX compiler
if(MSVC)
   set(msvc_common_compiler_flags -nologo -Oi -EHa- -GR- -bigobj -WX -wd4244 -Zc:__cplusplus)
   set(msvc_common_linker_flags "-nodefaultlib:msvcrt,-nodefaultlib:msvcrtd")
   string(REPLACE ";" " " cuda_host_common_compiler_flags "${msvc_common_compiler_flags}")
   string(REPLACE "," " " cuda_host_common_linker_flags "${msvc_common_linker_flags}")

   set_comp_flags(DEBUG   CXX  "${msvc_common_compiler_flags};-MTd")
   set_comp_flags(RELEASE CXX  "${msvc_common_compiler_flags};-MT")
   set_comp_flags(DEBUG   CUDA "-Xcompiler=${cuda_host_common_compiler_flags} -MTd")
   set_comp_flags(RELEASE CUDA "-Xcompiler=${cuda_host_common_compiler_flags} -MT")

   set_link_flags(DEBUG   CXX "LINKER:${msvc_common_linker_flags},-nodefaultlib:libcmt")
   set_link_flags(RELEASE CXX "LINKER:${msvc_common_linker_flags},-nodefaultlib:libcmtd,-OPT:REF,-OPT:ICF")
   set_link_flags(DEBUG   CUDA "-Xlinker=${cuda_host_common_linker_flags},-nodefaultlib:libcmt")
   set_link_flags(RELEASE CUDA "-Xlinker=${cuda_host_common_linker_flags},-nodefaultlib:libcmtd")

elseif((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
   set(clang_common_compiler_flags -Werror -Wall)
   set(clang_common_linker_flags "-static-libstdc++,-static-libgcc")

   set_comp_flags(DEBUG,RELEASE CXX,CUDA "${clang_common_compiler_flags}")
   set_link_flags(DEBUG,RELEASE CXX,CUDA "LINKER:${clang_common_linker_flags}")
endif()

set(cuda_common_compiler_flags --diag-suppress=20039,20040 --expt-relaxed-constexpr -Werror=all-warnings)
set(cuda_common_linker_flags "")
set_comp_flags(DEBUG,RELEASE CUDA "${cuda_common_compiler_flags}")
set_link_flags(DEBUG,RELEASE CUDA "LINKER:${cuda_common_linker_flags}")

# ---------- Dependencies ----------
if(HUBBARD_UPDATE_SUBMODULES)
   find_package(Git QUIET)
   if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
       execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       RESULT_VARIABLE GIT_SUBMOD_RESULT)
       if(NOT GIT_SUBMOD_RESULT EQUAL "0")
           message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}")
       endif()
   endif()
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenGL 3 REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("external/glfw")

target_include_directories(hubbard_compiler_flags INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/imgui>")
target_include_directories(hubbard_compiler_flags INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/implot>")

cmake_path(SET cuda_samples_path "$ENV{NVCUDASAMPLES_ROOT}/common/inc")
#include_directories(${cuda_samples_path})
target_include_directories(hubbard_compiler_flags INTERFACE "$<BUILD_INTERFACE:${cuda_samples_path}>")

# ---------- Targets ----------
include_directories("${PROJECT_SOURCE_DIR}/src")

# The code is built as a unity build except for hubbard_compute and ImGUI, which are separated into their own translation units
add_library(hubbard_compute OBJECT $<IF:${use_cuda},src/compute.cu,src/compute.cpp>)
#set_target_properties(hubbard_compute PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# NOTE: If, for some reason, files are added to/removed from the imgui or implot directories, have to reconfigure for the changes to be detected
file(GLOB imgui_sources
   "external/imgui/*.cpp" "external/imgui/*.h"
   "external/imgui/backends/imgui_impl_opengl3*"
   "external/imgui/backends/imgui_impl_glfw*"
   "external/implot/*.h"
   "external/implot/*.cpp")
add_library(imgui STATIC ${imgui_sources})
target_link_libraries(imgui PRIVATE hubbard_compiler_flags OpenGL::GL)
target_link_libraries(imgui PUBLIC glfw)

set(cusolver_libs CUDA::cusolver CUDA::cublas CUDA::cublasLt CUDA::cusparse CUDA::cuda_driver)
target_link_libraries(hubbard_compute PRIVATE hubbard_compiler_flags hubbard_linker_flags $<IF:${use_cuda},${cusolver_libs},Eigen3::Eigen>)

add_executable(hubbard src/hubbard.cpp)
target_link_libraries(hubbard PRIVATE hubbard_compiler_flags hubbard_linker_flags hubbard_compute Eigen3::Eigen imgui $<${WIN32}:winmm>)

# ---------- Testing ----------
if(HUBBARD_TESTING)
   enable_testing()
   include(GoogleTest)
   add_subdirectory("external/googletest")

   add_executable(hubbard_testing src/testing.cpp)

   #set_target_properties(gtest PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   #set_target_properties(gtest_main PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   #set_target_properties(hubbard_testing PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

   target_link_libraries(hubbard_testing PRIVATE hubbard_compiler_flags hubbard_linker_flags hubbard_compute Eigen3::Eigen gtest gmock gtest_main)

   set(gtest_args "--gtest_color=yes")
   gtest_discover_tests(hubbard_testing EXTRA_ARGS ${gtest_args}) # For use with ctest
   add_custom_target(test_hubbard COMMAND hubbard_testing ${gtest_args} DEPENDS hubbard_testing) # Without ctest
endif()

# ---------- Utilities ----------
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
   cmake_path(SET debug_exe_path "${CMAKE_BINARY_DIR}/Debug/${PROJECT_NAME}")
   cmake_path(SET release_exe_path "${CMAKE_BINARY_DIR}/Release/${PROJECT_NAME}")

   file(WRITE "${CMAKE_BINARY_DIR}/run.bat" "
@echo off 
${debug_exe_path}
")

   file(WRITE "${CMAKE_BINARY_DIR}/runrel.bat" "
@echo off 
${release_exe_path}
")
endif()
