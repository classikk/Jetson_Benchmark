#!/bin/bash

# URL for VS Code ARM64 .deb
URL="https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64"

# Output filename
FILE="vscode-arm64.deb"

echo "Downloading Visual Studio Code..."
curl -L "$URL" -o "$FILE"

echo "Installing Visual Studio Code..."
sudo apt install ./"$FILE"

echo "Done."

