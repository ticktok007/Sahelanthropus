#!/bin/bash
# Script to build QEMU from source with TCG plugins enabled for RISC-V 64-bit

echo "--- Installing build dependencies ---"
# Arch Linux dependencies for QEMU
sudo pacman -S --needed git base-devel ninja pkgconf glib2 python

echo "--- Cloning QEMU repository ---"
if [ ! -d "qemu" ]; then
    git clone https://gitlab.com/qemu-project/qemu.git
else
    echo "QEMU directory already exists. Skipping clone."
fi

cd qemu

echo "--- Configuring QEMU ---"
# Your requested configuration
./configure --enable-plugins --target-list=riscv64-linux-user

echo "--- Compiling QEMU ---"
# Compile using all available CPU cores to significantly speed up the build
make -j$(nproc)

echo "--- Build Complete! ---"
echo "You can find your compiled binary at: ./build/qemu-riscv64"
echo "Verify version:"
./build/qemu-riscv64 --version