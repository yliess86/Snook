FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
LABEL maintainer=yliess.hati@devinci.fr

ARG CYAN='\033[0;36m'
ARG NO_COLOR='\033[0m'

RUN echo "\n${CYAN}Updating & Installing Generic Dependencies${NO_COLOR}"
RUN DEBIAN_FRONTEND=noninteractive apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo build-essential wget git tar gzip subversion \
    parallel software-properties-common \
    libffi-dev libssl-dev libx11-dev libxxf86vm-dev libxcursor-dev \
    libxi-dev libxrandr-dev libxinerama-dev libglew-dev zlib1g-dev \
    libopenimageio-dev libopencolorio-dev libopenexr-dev \
    libopenjp2-7-dev libsndfile1-dev libfftw3-dev \
    opencollada-dev libjemalloc-dev libspnav-dev \
    libopenvdb-dev libblosc-dev libtbb-dev libceres-dev \
	libreadline-gplv2-dev libncursesw5-dev libsqlite3-dev tk-dev \
	libgdbm-dev libc6-dev libbz2-dev liblzma-dev

WORKDIR /
RUN echo "\n${CYAN}Installing Python3.7.9 from Source${NO_COLOR}"
ADD https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz Python-3.7.9.tgz
RUN tar xzf Python-3.7.9.tgz
RUN cd Python-3.7.9 && ./configure --enable-optimizations
RUN cd Python-3.7.9 && make install
RUN rm -rf Python-3.7.9

RUN echo "\n${CYAN}Installing and Updating Pip3${NO_COLOR}"
RUN DEBIAN_FRONTEND=noninteractive python3 -m pip install -U --upgrade pip setuptools wheel

RUN echo "\n${CYAN}Installing BPY Pip Module${NO_COLOR}"
RUN wget https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
RUN DEBIAN_FRONTEND=noninteractive python3 -m pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
RUN rm bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl

RUN echo "\n${CYAN}Cloning Snook${NO_COLOR};"
RUN git clone https://github.com/yliess86/Snook
WORKDIR /Snook

RUN echo "\n${CYAN}Installing Module Dependencies${NO_COLOR}"
RUN DEBIAN_FRONTEND=noninteractive python3 -m pip install \
    pytest pytest-cov \
    matplotlib==3.2.0 \
    kubernetes==11.0.0 \
    tqdm==4.43.0 \
    scipy==1.4.1 \
    opencv_python==4.2.0.34 \
    numpy==1.18.1 \
    kfp_server_api==1.0.4 \
    kfp_pipeline_spec==0.1.2 \
    mathutils==2.81.2 \
    splogger==0.1.4 \
    kfp==1.1.1 \
    Pillow==8.0.1 \
    https://download.pytorch.org/whl/cu102/torch-1.7.0-cp37-cp37m-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu102/torchvision-0.8.1-cp37-cp37m-linux_x86_64.whl

RUN echo "\n${CYAN}Copying Resources${NO_COLOR}"
COPY resources/hdri resources/hdri

RUN echo "\n${CYAN}Testing BPY Import and Running Pytest${NO_COLOR}"
RUN python3 -c "import bpy"
RUN python3 -m pytest -v --cov snook; exit 0

ENTRYPOINT ["/bin/bash"]