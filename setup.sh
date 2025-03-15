#!/bin/bash

echo "🔹 Step 1: Checking existing SQLite version"
sqlite3 --version || echo "SQLite is not installed or not found"

echo "🔹 Step 2: Downloading SQLite 3.35.0+"
wget https://www.sqlite.org/2021/sqlite-autoconf-3350000.tar.gz -O sqlite.tar.gz

if [ -f "sqlite.tar.gz" ]; then
    echo "✅ Download successful"
else
    echo "❌ Download failed"
    exit 1
fi

echo "🔹 Step 3: Extracting SQLite"
tar -xvzf sqlite.tar.gz

cd sqlite-autoconf-3350000 || { echo "❌ Extraction failed"; exit 1; }

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

echo "🔹 Step 8: Verifying SQLite installation"
sqlite3 --version || echo "❌ SQLite installation failed"

echo "🔹 Step 9: Cleanup"
cd ..
rm -rf sqlite-autoconf-3350000 sqlite.tar.gz

echo "✅ SQLite setup complete"
