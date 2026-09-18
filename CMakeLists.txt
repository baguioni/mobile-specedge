cmake_minimum_required(VERSION 3.14)
project(specedge_graph_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# Pin to a released tag so builds are reproducible; bump deliberately.
# 5202104b5 (2026-09-07): picks up ~10 ggml-hexagon commits over b10615
# (f280b2698), including the fully-async HTP backend (#26501) and MUL_MAT
# fusion (#28202) -- see mobile-specedge/experiments/09-11-26/npu-host-topk-results.md
# for the ~1.6x draft-round speedup this was already measured to give.
FetchContent_Declare(
  llama_cpp
GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
GIT_TAG        5202104b59ada9005db079eea43882a2b7bf5802
GIT_SHALLOW    TRUE
)

# Build only the llama/ggml libraries this project links against.
# LLAMA_BUILD_COMMON is ON (unlike TESTS/EXAMPLES/SERVER below): the batch
# client needs its `llama-common` target for common_chat_templates_apply()
# -- the Jinja engine that runs a model's own chat_template, as opposed to
# llama_chat_apply_template()'s fixed C reimplementation of a few known
# templates -- so that `client.reasoning` maps to enable_thinking correctly
# for any model (see src/script/client.cpp's apply_chat_template()).
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
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

# YAML parser for the batch client's config file (src/script/client.cpp).
# yaml-cpp 0.8.0 still ships cmake_minimum_required(VERSION 3.4), which
# CMake >= 4.0 rejects; this lets its (and only its) sub-configure proceed.
# llama.cpp and nlohmann_json declare their own >= 3.5 minimums, so this
# does not weaken their builds.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
FetchContent_Declare(
  yaml_cpp
GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
GIT_TAG        0.8.0
GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp nlohmann_json yaml_cpp)

# gRPC/Protobuf are consumed from the system install (see src/specedge_grpc/,
# pre-generated from specedge.proto with a matching protoc/grpc_cpp_plugin
# version) rather than FetchContent'd, since gRPC is a heavy from-source build.
#
# Ubuntu's libgrpc++-dev (1.16.1) does not ship CMake config files
# (gRPCConfig.cmake), only pkg-config .pc files, so gRPC is located via
# pkg-config rather than find_package(gRPC CONFIG).
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

add_library(graph_engine_lib
src/graph_engine.cpp
src/host_topk.cpp
src/grpc_client.cpp
src/oracle_validator.cpp
src/tree.cpp
src/spec_exec_client.cpp
src/proactive_draft.cpp
src/topk_sampler.cpp
src/specedge_grpc/specedge.pb.cc
src/specedge_grpc/specedge.grpc.pb.cc
)
target_include_directories(graph_engine_lib PUBLIC src ${GRPC_INCLUDE_DIRS})
target_link_directories(graph_engine_lib PUBLIC ${GRPC_LIBRARY_DIRS})
target_compile_options(graph_engine_lib PUBLIC ${GRPC_CFLAGS_OTHER})
target_link_libraries(graph_engine_lib PUBLIC Threads::Threads llama nlohmann_json::nlohmann_json protobuf::libprotobuf ${GRPC_LIBRARIES})

# Local smoke test: a linear greedy decode, a tree-mode commit exactness
# check, and a SpecExecClient run against an in-process oracle. No target
# server, no gRPC round-trip -- enough to prove the llama.cpp build links and
# that a draft model (hybrid ones like Qwen3.5 included) runs through both the
# linear and tree paths on this machine (see src/local_test.cpp).
add_executable(local_test src/local_test.cpp)
target_link_libraries(local_test PRIVATE graph_engine_lib)

# Config-driven batch client: reads a YAML config, walks a whole dataset,
# runs each prompt through tree-based (SpecExec) speculative decoding. Port
# of specedge/src/script/client.py.
add_executable(client src/script/client.cpp)
target_link_libraries(client PRIVATE graph_engine_lib yaml-cpp::yaml-cpp llama-common)
# Regression test for Tree::reorder_by_sequence_proactive -- the slot remap a
# winning proactive bet goes through (see src/proactive_draft.h). Tree touches
# llama.h only for its typedefs, so this links without libllama, gRPC or a
# target server, which is what makes the splice cheap to check. Not built by
# default: -DSPECEDGE_BUILD_TESTS=ON, then ./build/tree_splice_test.
option(SPECEDGE_BUILD_TESTS "Build the unit tests" OFF)
if(SPECEDGE_BUILD_TESTS)
  # Guarded by EXISTS: this file is referenced but not present in this
  # checkout, and SPECEDGE_BUILD_TESTS should not fail to configure over a
  # test that isn't there rather than one that is broken.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test/tree_splice_test.cpp")
    add_executable(tree_splice_test src/test/tree_splice_test.cpp src/tree.cpp)
    target_include_directories(tree_splice_test PRIVATE src)
    target_link_libraries(tree_splice_test PRIVATE llama)
    enable_testing()
    add_test(NAME tree_splice COMMAND tree_splice_test)
  endif()

  # Tier-2 sandbox for ProactiveDraft's frontier-leaf fix (see
  # src/proactive_draft.cpp's FrontierLeaves/ChooseBet and
  # src/test/proactive_draft_test.cpp): a real LlamaCppEngine + GGUF model,
  # proving llama.cpp's KV cache tolerates ChooseBet's scratch-fork path.
  # Deliberately built from the leanest source set that provides
  # LlamaCppEngine -- graph_engine.cpp only needs nlohmann_json and llama,
  # not gRPC/protobuf/yaml-cpp -- rather than linking graph_engine_lib
  # (which pulls in the gRPC client/server plumbing this test never uses).
  add_executable(proactive_draft_test
    src/test/proactive_draft_test.cpp
    src/graph_engine.cpp
    src/host_topk.cpp
    src/topk_sampler.cpp
    src/tree.cpp
    src/proactive_draft.cpp
  )
  target_include_directories(proactive_draft_test PRIVATE src)
  target_link_libraries(proactive_draft_test PRIVATE llama nlohmann_json::nlohmann_json)
  target_compile_definitions(proactive_draft_test PRIVATE
    PROACTIVE_DRAFT_TEST_MODEL_PATH="${CMAKE_SOURCE_DIR}/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf")
  enable_testing()
  add_test(NAME proactive_draft COMMAND proactive_draft_test)
endif()
