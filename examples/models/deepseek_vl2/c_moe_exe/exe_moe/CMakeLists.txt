cmake_minimum_required(VERSION 3.18)
project(deepseek_moe_exe C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Torch REQUIRED)  

set(ET_PY_INCLUDE "" CACHE PATH "ExecuTorch pip include override")
set(ET_PY_LIB     "" CACHE FILEPATH "ExecuTorch pip lib override")
set(ET_PY_CORE_LIB "" CACHE FILEPATH "ExecuTorch core shared lib override")

if(NOT ET_PY_INCLUDE OR NOT ET_PY_LIB)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import importlib.util, os, glob;
spec = importlib.util.find_spec('executorch');
assert spec and spec.submodule_search_locations, 'executorch not found or no search locations';
root = spec.submodule_search_locations[0];
inc  = os.path.join(root, 'include');
cands = glob.glob(os.path.join(root, 'lib', 'libexecutorch*.so'));
print(inc); print(cands[0] if cands else '')"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE ET_LINES
  )
  string(REPLACE "\n" ";" ET_LIST ${ET_LINES})
  list(LENGTH ET_LIST _len)
  if(_len GREATER 0)
    list(GET ET_LIST 0 ET_PY_INCLUDE)
  endif()
  if(_len GREATER 1)
    list(GET ET_LIST 1 ET_PY_LIB)
  endif()
endif()

get_filename_component(ET_PY_LIBDIR ${ET_PY_LIB} DIRECTORY)

if(NOT ET_PY_CORE_LIB)
  set(ET_PY_CORE_LIB "${ET_PY_LIBDIR}/libexecutorch_core.so")
endif()

message(STATUS "ET_PY_INCLUDE='${ET_PY_INCLUDE}'")
message(STATUS "ET_PY_LIB='${ET_PY_LIB}'")
message(STATUS "ET_PY_CORE_LIB='${ET_PY_CORE_LIB}'")

if (NOT ET_PY_INCLUDE OR NOT EXISTS ${ET_PY_INCLUDE})
  message(FATAL_ERROR "ExecuTorch pip include not found: ${ET_PY_INCLUDE}")
endif()
if (NOT ET_PY_LIB OR NOT EXISTS ${ET_PY_LIB})
  message(FATAL_ERROR "ExecuTorch pip lib not found: ${ET_PY_LIB}")
endif()

add_library(deepseek_moe_execu SHARED
  ${CMAKE_CURRENT_LIST_DIR}/moe_kernel.cpp
)

target_compile_definitions(deepseek_moe_execu PRIVATE EXECUTORCH_ENABLE_LOGGING=0)

target_include_directories(deepseek_moe_execu PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../common
  ${ET_PY_INCLUDE}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(deepseek_moe_execu PRIVATE
  ${ET_PY_LIB}
  ${ET_PY_CORE_LIB}
)

target_link_options(deepseek_moe_execu PRIVATE "-Wl,-rpath,${ET_PY_LIBDIR}")

set_target_properties(deepseek_moe_execu PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS deepseek_moe_execu
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)