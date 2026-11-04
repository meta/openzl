# Copyright (c) Meta Platforms, Inc. and affiliates.

# Define zip_lexer library
add_library(zip_lexer
    zip_lexer.c
)
target_include_directories(zip_lexer PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR})
target_link_libraries(zip_lexer
    openzl
)
add_dependencies(zip_lexer openzl)
apply_openzl_compile_options_to_target(zip_lexer)

# Define pytorch_model_parser library
add_library(pytorch_model_parser
    pytorch_model_parser.c
)
target_include_directories(pytorch_model_parser PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR})
target_link_libraries(pytorch_model_parser
    zip_lexer
    openzl
)
add_dependencies(pytorch_model_parser zip_lexer openzl)
apply_openzl_compile_options_to_target(pytorch_model_parser)

# Add subdirectories
add_subdirectory(csv)
add_subdirectory(gguf)
add_subdirectory(numpy)
add_subdirectory(onnx)
add_subdirectory(parquet)
add_subdirectory(safetensors)
add_subdirectory(sddl)
add_subdirectory(shared_components)

# Add the custom_parsers library
add_library(custom_parsers
    dependency_registration.cpp
)
target_include_directories(custom_parsers PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR})
target_link_libraries(custom_parsers
    openzl
    openzl_cpp
    csv_parser
    gguf_parser
    numpy_parser
    onnx_parser
    parquet_graph
    safetensors_parser
    shared_components
)
add_dependencies(custom_parsers openzl openzl_cpp)
apply_openzl_compile_options_to_target(custom_parsers
    csv_parser
    gguf_parser
    numpy_parser
    onnx_parser
    parquet_graph
    safetensors_parser
    shared_components
)
