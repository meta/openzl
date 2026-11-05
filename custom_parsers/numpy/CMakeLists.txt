# Copyright (c) Meta Platforms, Inc. and affiliates.

file(
    GLOB numpy_sources
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.c")
file(
    GLOB numpy_parser_headers
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/*.h")

add_library(numpy_parser ${numpy_sources})
target_include_directories(numpy_parser PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR})
target_link_libraries(numpy_parser
    openzl
    pickle_lexer
    shared_components
    zip_lexer
)

add_dependencies(numpy_parser openzl)
apply_openzl_compile_options_to_target(numpy_parser)
