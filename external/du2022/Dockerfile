# Reference implementation of Du et al. 2022, built for the head-to-head
# comparison.
#
# Ubuntu 22.04 for GCC 11, because the code and its Abseil pin are from 2022 and
# GCC 13 rejects them. But the distro's CMake 3.22 is too old for the project's
# use of policy CMP0135, so CMake comes from Kitware's release tarball instead.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=3.31.6

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
      | tar -xz -C /opt \
    && ln -s "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake \
    && cmake --version

WORKDIR /opt
RUN git clone --depth 1 https://github.com/duxingyi-charles/Robust-Implicit-Surface-Networks.git

WORKDIR /opt/Robust-Implicit-Surface-Networks/build

# The code and its dependencies rely on headers that libstdc++ used to pull in
# transitively and no longer does, so force-include them everywhere rather than
# editing their sources. -Wno-error because the build treats warnings as errors
# and the pinned nlohmann_json uses std::is_pod, deprecated since C++20.
RUN cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wno-error -Wno-deprecated-declarations \
          -include memory -include limits -include cstdint \
          -include algorithm -include array -include string" ..

# Their loader only evaluates analytic primitives, so there is no way to hand it
# a field that has no closed form. This adds one branch that reads per-vertex
# values from a file and writes them into the same column of the same matrix
# every other branch fills; the arrangement code downstream is untouched.
COPY sampled-values.patch /tmp/sampled-values.patch
RUN git -C _deps/implicit_functions-src apply /tmp/sampled-values.patch

RUN make -j"$(nproc)" material_interface

WORKDIR /work
ENTRYPOINT ["/opt/Robust-Implicit-Surface-Networks/build/material_interface"]
