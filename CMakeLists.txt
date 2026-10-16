cmake_minimum_required(VERSION 3.21)
project(PLeXQuantizations VERSION 1.0.0 LANGUAGES CXX)
set(MLN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../MyeLiN" CACHE PATH
    "Path to the MyeLiN CMake source tree")

include(CTest)
if(NOT TARGET mln_sdk)
  set(MLN_BUILD_RUNTIME OFF CACHE BOOL "" FORCE)
  set(MLN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${MLN_ROOT}" MyeLiN)
endif()

add_executable(quantizations_codecs_test tests/codecs_test.cpp)
target_compile_features(quantizations_codecs_test PRIVATE cxx_std_23)
target_include_directories(quantizations_codecs_test PRIVATE
  .shared/include
  Q4K/include
  Q5K/include
  Q6K/include
  Q8/include
)
target_link_libraries(quantizations_codecs_test PRIVATE mln_sdk)

add_test(NAME quantizations_codecs COMMAND quantizations_codecs_test)

add_executable(quantizations_decode_graph_test tests/decode_graph_test.cpp)
target_compile_features(quantizations_decode_graph_test PRIVATE cxx_std_23)
target_include_directories(quantizations_decode_graph_test PRIVATE
  .shared/include
  Q4K/include
  Q5K/include
  Q6K/include
  Q8/include
)
target_link_libraries(quantizations_decode_graph_test PRIVATE mln_sdk)
add_test(NAME quantizations_decode_graph
  COMMAND quantizations_decode_graph_test)

set(SYN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../SYNaPSe" CACHE PATH
    "Path to SYNaPSe public compiler contracts")
add_executable(quantizations_compatibility_test tests/compatibility_test.cpp)
target_compile_features(quantizations_compatibility_test PRIVATE cxx_std_23)
target_include_directories(quantizations_compatibility_test PRIVATE
    .shared/include Q8/include "${SYN_ROOT}/include")
target_link_libraries(quantizations_compatibility_test PRIVATE mln_sdk)
add_test(NAME quantizations_compatibility COMMAND quantizations_compatibility_test)
