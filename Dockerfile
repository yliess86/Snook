FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer=yliess.hati@devinci.fr

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget python3.8 python3.8-dev python3.8-distutils
RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
RUN rm get-pip.py
RUN DEBIAN_FRONTEND=noninteractive pip3 install --upgrade pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo build-essential git tar gzip subversion \
    libssl-dev libx11-dev libxxf86vm-dev libxcursor-dev \
    libxi-dev libxrandr-dev libxinerama-dev libglew-dev \
    libopenimageio-dev libopencolorio-dev libopenexr-dev \
    libopenjp2-7-dev libsndfile1-dev libfftw3-dev \
    opencollada-dev libjemalloc-dev libspnav-dev \
    libopenvdb-dev libblosc-dev libtbb-dev libceres-dev

WORKDIR /
RUN DEBIAN_FRONTEND=noninteractive apt --purge remove cmake -y
RUN curl -s "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

RUN git clone https://github.com/blender/blender
WORKDIR /blender
RUN git submodule update --init --recursive
RUN git submodule foreach git checkout master
RUN git submodule foreach git pull --rebase origin master

ENV DEPS=build_files/build_environment/install_deps.sh
RUN sed -i 's/PYTHON_VERSION="3.7.7"/PYTHON_VERSION="3.8.0"/g' ${DEPS}
RUN sed -i 's/PYTHON_VERSION_SHORT="3.7"/PYTHON_VERSION_SHORT="3.8"/g' ${DEPS}
RUN sed -i 's/PYTHON_VERSION_MIN="3.7"/PYTHON_VERSION_MIN="3.8"/g' ${DEPS}

RUN bash build_files/build_environment/install_deps.sh --with-embree --build-embree --skip-python --no-confirm
RUN mkdir /blender/build

WORKDIR /blender/build
RUN cmake \
    -D WITH_PYTHON_INSTALL=OFF \
    -D WITH_AUDASPACE=OFF \
    -D WITH_PYTHON_MODULE=ON \
    -D WITH_INSTALL_PORTABLE=OFF \
    -D WITH_CYCLES_CUDA_BINARIES=ON \
    -D PYTHON_LIBRARY=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so \
    -D PYTHON_INCLUDE_DIR=/usr/include/python3.8 \
    -D CMAKE_INSTALL_PREFIX=/usr/local/lib/python3.8/site-packages \
    -D PYTHON_VERSION=3.8 \
    ..
RUN make -j 8 && make install
ENV PYTHONPATH=/usr/local/lib/python3.8/dist-packages/

WORKDIR /
RUN rm -rf blender

RUN git clone https://github.com/yliess86/Snook
WORKDIR /Snook
COPY resources/* resources/
RUN DEBIAN_FRONTEND=noninteractive pip3 install -r requirements.txt

ENTRYPOINT [ "bash" ]