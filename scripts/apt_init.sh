#!/usr/bin/env bash

sudo apt install -y git clang
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/AllTheLines/CacheTVS
cd CacheTVS && git checkout cpu-clean
