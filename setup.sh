#!/bin/bash

echo "🔹 Step 1: Checking existing SQLite version"
sqlite3 --version || echo "SQLite is not installed or not found"

echo "🔹 Step 2: Downloading SQLite 3.39.2 (latest stable)"
wget https://www.sqlite.org/2022/sqlite-autoconf-3390200.tar.gz -O sqlite.tar.gz

if [ -f "sqlite.tar.gz" ]; then
    echo "✅ Download successful"
else
    echo "❌ Download failed"
    exit 1
fi

echo "🔹 Step 3: Extracting SQLite"
tar -xvzf sqlite.tar.gz

cd sqlite-autoconf-3390200 || { echo "❌ Extraction failed"; exit 1; }

echo "🔹 Step 4: Configuring SQLite"
./configure --prefix=$HOME/.local || { echo "❌ Configure failed"; exit 1; }

echo "🔹 Step 5: Compiling SQLite (this may take a while)"
make -j$(nproc) || { echo "❌ Compilation failed"; exit 1; }

echo "🔹 Step 6: Installing SQLite"
make install || { echo "❌ Installation failed"; exit 1; }

echo "🔹 Step 7: Updating library paths"
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
source ~/.bashrc

echo "🔹 Step 8: Checking new SQLite version"
sqlite3 --version || echo "❌ SQLite installation failed"

echo "🔹 Step 9: Reinstalling Python to Use New SQLite"
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-venv libsqlite3-dev

echo "✅ SQLite setup complete"
