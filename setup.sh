#!/bin/bash

# Download and compile SQLite 3.35.0+
wget https://www.sqlite.org/2021/sqlite-autoconf-3350000.tar.gz
tar -xvzf sqlite-autoconf-3350000.tar.gz
cd sqlite-autoconf-3350000

# Configure and install SQLite
./configure --prefix=$HOME/.local
make
make install

# Update LD_LIBRARY_PATH so Python uses the new SQLite version
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Clean up
cd ..
rm -rf sqlite-autoconf-3350000 sqlite-autoconf-3350000.tar.gz

# Verify SQLite version
sqlite3 --version
