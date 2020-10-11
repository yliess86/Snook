# Install Blender Lib Dependencies
pip3 install --upgrade pip
apt-get install -y \
    sudo build-essential git tar gzip subversion \
    libssl-dev libx11-dev libxxf86vm-dev libxcursor-dev \
    libxi-dev libxrandr-dev libxinerama-dev libglew-dev \
    libopenimageio-dev libopencolorio-dev libopenexr-dev \
    libopenjp2-7-dev libsndfile1-dev libfftw3-dev \
    opencollada-dev libjemalloc-dev libspnav-dev \
    libopenvdb-dev libblosc-dev libtbb-dev libceres-dev

# Upgrade CMake
apt --purge remove cmake -y
curl -s "https://cmake.org/files/v3.18/cmake-3.18.2-Linux-x86_64.tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

# Recursiverly Clones Blender
git clone https://github.com/blender/blender
cd blender
git submodule update --init --recursive
git submodule foreach git checkout master
git submodule foreach git pull --rebase origin master

# Replace Python 3.7 by 3.8 in Blender Scripts
DEPS=build_files/build_environment/install_deps.sh
sed -i 's/PYTHON_VERSION="3.7.7"/PYTHON_VERSION="3.8.0"/g' ${DEPS}
sed -i 's/PYTHON_VERSION_SHORT="3.7"/PYTHON_VERSION_SHORT="3.8"/g' ${DEPS}
sed -i 's/PYTHON_VERSION_MIN="3.7"/PYTHON_VERSION_MIN="3.8"/g' ${DEPS}

# Build and Install Blender Dependencies
bash build_files/build_environment/install_deps.sh --with-embree --build-embree --skip-python --no-confirm

# Compile and Install Blender
mkdir build
cd build
cmake \
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
make -j 8 && make install

# Remove Blender Sources
cd ../..
rm -rf blender