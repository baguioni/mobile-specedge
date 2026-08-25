cmake_minimum_required(VERSION 3.14)
project(specedge_graph_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Pin to a released tag so builds are reproducible; bump deliberately.
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        b10615
  GIT_SHALLOW    TRUE
)

# Build only the llama/ggml libraries this project links against.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp nlohmann_json)

# gRPC/Protobuf are consumed from the system install (see src/specedge_grpc/,
# pre-generated from specedge.proto with a matching protoc/grpc_cpp_plugin
# version) rather than FetchContent'd, since gRPC is a heavy from-source build.
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

add_library(graph_engine_lib
  src/graph_engine.cpp
  src/grpc_client.cpp
  src/chain.cpp
  src/spec_client.cpp
  src/specedge_grpc/specedge.pb.cc
  src/specedge_grpc/specedge.grpc.pb.cc
)
target_include_directories(graph_engine_lib PUBLIC src)
target_link_libraries(graph_engine_lib PUBLIC llama nlohmann_json::nlohmann_json protobuf::libprotobuf gRPC::grpc++)

add_executable(graph_engine src/graph_engine_main.cpp)
target_link_libraries(graph_engine PRIVATE graph_engine_lib)

add_executable(parity_test src/parity_test.cpp)
target_link_libraries(parity_test PRIVATE graph_engine_lib)

add_executable(linear_client src/client_main.cpp)
target_link_libraries(linear_client PRIVATE graph_engine_lib)
