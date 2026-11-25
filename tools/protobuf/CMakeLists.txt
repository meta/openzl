if (OPENZL_BUILD_PROTOBUF_TOOLS)
    cmake_minimum_required(VERSION 3.24)
    # Configure protobuf options before fetching
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_WITH_ZLIB_DEFAULT OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    # Get the protobuf library
    FetchContent_Declare(protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG        v32.0
        FIND_PACKAGE_ARGS NAMES protobuf
    )
    FetchContent_MakeAvailable(protobuf)
    if(TARGET libprotobuf AND NOT TARGET protobuf::libprotobuf)
        target_compile_options(libprotobuf PUBLIC
            -Wno-deprecated-declarations
            -Wno-stringop-overread
            -Wno-strict-aliasing
            -Wno-attributes
        )
    endif()
    find_package(protobuf CONFIG REQUIRED)

    # Make the protobuf_serializer library
    file(GLOB protobuf_serializer_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    )
    list(REMOVE_ITEM protobuf_serializer_sources "${CMAKE_CURRENT_SOURCE_DIR}/cli.cpp")

    add_library(protobuf_serializer ${protobuf_serializer_sources})

    target_link_libraries(protobuf_serializer PUBLIC
        openzl_cpp
        protobuf::libprotobuf
        protobuf::libprotoc
    )
    apply_openzl_compile_options_to_target(protobuf_serializer)

    # Make the protobuf_cli tool
    file(GLOB protobuf_cli_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cli.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/schema.proto")
    add_executable(protobuf_cli ${protobuf_cli_sources})

    include(FindProtobuf)
    protobuf_generate(TARGET protobuf_cli)

    target_link_libraries(protobuf_cli PRIVATE
        protobuf_serializer
        arg
        tools_io
        tools_training
        custom_parsers
    )
    target_include_directories(protobuf_cli PRIVATE ${PROJECT_BINARY_DIR})

    apply_openzl_compile_options_to_target(protobuf_cli)

    # Build tests if enabled
    if(OPENZL_BUILD_TESTS)
        add_library(protobuf_test_utils "${CMAKE_CURRENT_SOURCE_DIR}/tests/utils.cpp")
        target_include_directories(protobuf_test_utils PRIVATE
            ${PROJECT_SOURCE_DIR})

        # Test for DescriptorLoader
        add_executable(test_protobuf_descriptor_loader
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_descriptor_loader.cpp")
        target_link_libraries(test_protobuf_descriptor_loader PRIVATE
            protobuf_serializer
            GTest::gtest_main
            protobuf_test_utils
        )
        gtest_discover_tests(test_protobuf_descriptor_loader)

        # Test for DynamicMessageHelper
        add_executable(test_protobuf_dynamic_messages
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dynamic_messages.cpp")
        target_link_libraries(test_protobuf_dynamic_messages PRIVATE
            protobuf_serializer
            GTest::gtest_main
            protobuf_test_utils
        )
        gtest_discover_tests(test_protobuf_dynamic_messages)

        # Test for ProtoSerializer and ProtoDeserializer
        add_executable(test_proto_serializer
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_proto_serializer.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_schema.proto")
        protobuf_generate(TARGET test_proto_serializer)
        target_include_directories(test_proto_serializer PRIVATE
            ${PROJECT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
        )
        target_link_libraries(test_proto_serializer PRIVATE
            protobuf_serializer
            custom_parsers
            tools_training
            GTest::gtest_main
        )
        gtest_discover_tests(test_proto_serializer)
    endif()
endif()
