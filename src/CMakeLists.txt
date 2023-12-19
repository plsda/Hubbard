
cmake_minimum_required(VERSION 3.25)

project(Hubbard LANGUAGES CXX)

option(FAST_FP OFF)
option(HUBBARD_TESTING "$<STREQUAL:${CMAKE_PROJECT_NAME}, ${PROJECT_NAME}>")

# Build options #
add_library(hubbard_compiler_flags INTERFACE)
add_library(hubbard_linker_flags INTERFACE)

target_compile_features(hubbard_compiler_flags INTERFACE cxx_std_20)
target_compile_definitions(hubbard_compiler_flags INTERFACE "$<$<CONFIG:DEBUG>:HUBBARD_DEBUG>")

set(cxx_msvc_like  $<AND:$<COMPILE_LANGUAGE:CXX>, $<OR:$<LANG_COMPILER_ID:MSVC>, $<STREQUAL:${CMAKE_CXX_SIMULATE_ID}, "MSVC" >>>)
set(cxx_clang_like $<AND:$<COMPILE_LANGUAGE:CXX>, $<OR:$<LANG_COMPILER_ID:MSVC>, $<STREQUAL:${CMAKE_CXX_SIMULATE_ID}, "Clang" >>>)

if(MSVC)
   set(msvc_common_compiler_flags "-nologo;-Oi;-EHa-;-GR-;$<$<BOOL:${FAST_FP}>:-fp:fast;>")#,-openmp")
   set(msvc_common_linker_flags "")

   target_compile_options(hubbard_compiler_flags INTERFACE 
      #"$<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:${msvc_common_compiler_flags}-MTd;-Wall;-WX;-wd4244>>"
      "$<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:${msvc_common_compiler_flags}-MTd;-W4;-WX;-wd4244>>"
      "$<$<CONFIG:RELEASE>:$<BUILD_INTERFACE:${msvc_common_compiler_flags}-MT>>")
   
   target_link_options(hubbard_linker_flags INTERFACE
      "$<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:LINKER:${msvc_common_linker_flags}>>"
      "$<$<CONFIG:RELEASE>:$<BUILD_INTERFACE:LINKER:${msvc_common_linker_flags}-OPT:REF,-OPT:ICF>>") 
elseif(${cxx_clang_like})
   set(clang_common_compiler_flags "")
   set(clang_common_linker_flags "")
endif()

find_package(Eigen3 3.4 REQUIRED)

add_executable(hubbard hubbard.cpp)
target_link_libraries(hubbard hubbard_compiler_flags Eigen3::Eigen)

# Utilities #
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
