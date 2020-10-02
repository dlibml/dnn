#!/usr/bin/env bash

BUILD_PREFIX=build

ARG1=`echo $1 | awk '{print tolower($0)}'`

BUILD_TYPE=Release

case ${ARG1} in
    release|release/)
        BUILD_TYPE=Release
        shift 1
        ;;
    debug|debug/)
        BUILD_TYPE=Debug
        shift 1
        ;;
    *)
        BUILD_TYPE=Release
        ;;
esac

cmake -B ${BUILD_PREFIX}/${BUILD_TYPE} -GNinja \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DUSE_AVX_INSTRUCTIONS=ON \
    -DUSE_SSE2_INSTRUCTIONS=ON \
    -DUSE_SSE4_INSTRUCTIONS=ON \
    ${@}

cmake --build ${BUILD_PREFIX}/${BUILD_TYPE} --config ${BUILD_TYPE}
