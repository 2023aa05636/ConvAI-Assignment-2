#!/bin/bash

echo "✅ Running setup.sh script..." > setup_log.txt

echo "🔹 Checking existing SQLite version"
sqlite3 --version || echo "SQLite is not installed or not found"

echo "🔹 Updating package list"
apt-get update

echo "🔹 Installing required dependencies"
apt-get install -y libsqlite3-dev build-essential python3-dev > /dev/null 2>&1

echo "🔹 Downloading and installing SQLite 3.39.2"
wget https://www.sqlite.org/2022/sqlite-autoconf-3390200.tar.gz -O sqlite.tar.gz
tar -xvzf sqlite.tar.gz
cd sqlite-autoconf-3390200
./configure --prefix=$HOME/.local --enable-load-extension
make -j$(nproc)
make install

echo "🔹 Updating library paths"
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc

echo "✅ setup.sh execution complete." >> setup_log.txt
