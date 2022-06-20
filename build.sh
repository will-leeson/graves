#!/bin/bash

pushd src/graph-builder
pwd

if [ ! -d "src/graph-builder/build" ]
then
    mkdir build
else
    rm -r build
    mkdir build
fi

pushd build

if command -v cmake &> /dev/null
then
    if command -v ninja &> /dev/null
    then
        cmake -G Ninja ..
        ninja
    elif command -v make &> /dev/null
    then
        cmake ..
        make
    else
        echo "I can't find ninja or make. Please install one and re-attempt the build process"
    fi
else
    echo "I can't find CMake. Please install CMake and re-attempt the build process"
    exit 1
fi

popd

if [ -d "${HOME}/.local/bin" ]
then
    ln -s $(pwd)/build/src/graph-builder ${HOME}/.local/bin
else
    mkdir "${HOME}/.local/bin"
    ln -s $(pwd)/build/src/graph-builder ${HOME}/.local/bin
fi

echo "Make sure to add \"${HOME}/.local/bin\" to your PATH"

popd