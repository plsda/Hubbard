
cmake_minimum_required(VERSION 3.25)

project(Hubbard LANGUAGES CXX)

option(FAST_FP OFF)
option(HUBBARD_TESTING "$<STREQUAL:${CMAKE_PROJECT_NAME}, ${PROJECT_NAME}>")

# Build options #
add_library(hubbard_compiler_flags INTERFACE)
add_library(hubbard_linker_flags INTERFACE)

target_compile_features(hubbard_compiler_flags INTERFACE cxx_std_20)
target_compile_definitions(hubbard_compiler_flags INTERFACE "$<$<CONFIG:DEBUG>:HUBBARD_DEBUG>")

set(cxx_msvc_like  $<OR:$<CXX_COMPILER_ID:MSVC>, $<STREQUAL:${CMAKE_CXX_SIMULATE_ID},MSVC>>)
set(cxx_clang_like $<OR:$<CXX_COMPILER_ID:Clang>, $<STREQUAL:${CMAKE_CXX_SIMULATE_ID},Clang>>)

if(MSVC)
   set(msvc_common_compiler_flags "-nologo;-Oi;-EHa-;-GR-;$<$<BOOL:${FAST_FP}>:-fp:fast;>")
   set(msvc_common_linker_flags "")

   target_compile_options(hubbard_compiler_flags INTERFACE 
      "$<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:${msvc_common_compiler_flags}-MTd;-WX;-wd4244>>"
      "$<$<CONFIG:RELEASE>:$<BUILD_INTERFACE:${msvc_common_compiler_flags}-MT>>")
   
   target_link_options(hubbard_linker_flags INTERFACE
      "$<$<CONFIG:DEBUG>:$<BUILD_INTERFACE:LINKER:${msvc_common_linker_flags}>>"
      "$<$<CONFIG:RELEASE>:$<BUILD_INTERFACE:LINKER:${msvc_common_linker_flags}-OPT:REF,-OPT:ICF>>") 
elseif((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
   set(clang_common_compiler_flags "-Werror;-Wall;") # -Wextra
   set(clang_common_linker_flags "-static-libstdc++,-static-libgcc")

   target_compile_options(hubbard_compiler_flags INTERFACE 
      "$<BUILD_INTERFACE:${clang_common_compiler_flags}>")

   target_link_options(hubbard_linker_flags INTERFACE
      "$<BUILD_INTERFACE:LINKER:${clang_common_linker_flags}>")
endif()

find_package(Eigen3 3.4 REQUIRED)

include_directories("${PROJECT_SOURCE_DIR}/src")
add_executable(hubbard src/hubbard.cpp)
target_link_libraries(hubbard hubbard_compiler_flags hubbard_linker_flags Eigen3::Eigen)


# Testing # 
if(HUBBARD_TESTING) 
   #set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # If using shared CRT (currently using static)
   enable_testing()
   include(GoogleTest)
   add_subdirectory("external/googletest")

   add_executable(hubbard_testing src/testing.cpp)

   set_target_properties(gtest PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   set_target_properties(gtest_main PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   set_target_properties(hubbard_testing PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

   target_link_libraries(hubbard_testing hubbard_compiler_flags hubbard_linker_flags Eigen3::Eigen gtest gmock gtest_main)

   gtest_discover_tests(hubbard_testing EXTRA_ARGS --output-on-failure --gtest_color=yes)
endif()

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
