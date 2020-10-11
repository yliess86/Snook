URL=https://github.com/kendryte/nncase/releases/download/v0.2.0-beta4
FILE=ncc_linux_x86_64.tar.xz
DEST=./vendor/bin

mkdir ${DEST}
wget -O ${DEST}/${FILE} ${URL}/${FILE}
tar xf ${DEST}/${FILE} -C ${DEST}
rm ${DEST}/${FILE}