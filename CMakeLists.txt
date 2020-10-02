cmake_minimum_required(VERSION 3.14)
project("dlib template project" LANGUAGES CXX)

set(CPACK_PACKAGE_NAME "dlib-template-project")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

# Use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Colored warnings
option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." ON)
if(${FORCE_COLORED_OUTPUT})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options (-fdiagnostics-color=always)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options (-fcolor-diagnostics)
    endif()
endif()

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

# Optimization flags
include(CheckCXXCompilerFlag)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Dependency management
include(FetchContent)
macro(fetch_content name tag repository)
    FetchContent_Declare(
        ${name}
        GIT_REPOSITORY ${repository}
        GIT_TAG        ${tag}
        GIT_PROGRESS   TRUE
        USES_TERMINAL_DOWNLOAD TRUE
        SOURCE_DIR     ${PROJECT_SOURCE_DIR}/external/${name}
    )
    message("-- Fetching ${name} ${tag}")
    FetchContent_MakeAvailable(${name})
endmacro()

macro(add_dlib_executable name)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} PRIVATE dlib::dlib)
    target_include_directories(${name} PRIVATE src)
    target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic -Wno-deprecated-copy)
    install(TARGETS ${name} DESTINATION bin)
endmacro()

fetch_content(dlib master https://github.com/davisking/dlib.git)

add_dlib_executable(main)
