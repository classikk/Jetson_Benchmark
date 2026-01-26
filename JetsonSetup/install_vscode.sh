#!/bin/bash

# URL for VS Code ARM64 .deb
URL="https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64"
FILE="vscode-arm64.deb"

# Disable Microsoft repo auto‑add
sudo mkdir -p /etc/vscode
echo "false" | sudo tee /etc/vscode/disable-repo > /dev/null

echo "Downloading Visual Studio Code..."
curl -L "$URL" -o "$FILE"

echo "Installing Visual Studio Code without adding Microsoft repo..."
sudo apt install ./"$FILE"

echo "Done."
