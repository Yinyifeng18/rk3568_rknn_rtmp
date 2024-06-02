set -e

TARGET_SOC="rk356x"
GCC_COMPILER=/opt/atk-dlrk356x-toolchain/usr/bin/aarch64-buildroot-linux-gnu

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
# build
BUILD_DIR=${ROOT_PWD}/build/

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ..
make -j4
make install
cd -
