#!/usr/bin/env bash
# Install OpenJDK 21 to libs/openjdk/ (project-local, no root required).
# Works on Linux (x64, aarch64) and macOS (x64, aarch64).
#
# Usage:
#   bash scripts/install_java.sh
#
# After running, set these environment variables before using Java/PyTerrier:
#   export JAVA_HOME="$(pwd)/libs/openjdk"
#   export JVM_PATH="$(find libs/openjdk -name 'libjvm.*' | head -1)"
#   export PATH="$JAVA_HOME/bin:$PATH"

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)/libs/openjdk"
JDK_VERSION="21"

# --- Detect platform ---
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="mac" ;;
    *)      echo "ERROR: Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64)  ARCH_TAG="x64" ;;
    aarch64|arm64) ARCH_TAG="aarch64" ;;
    *)             echo "ERROR: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# --- Check if already installed ---
if [ -x "$INSTALL_DIR/bin/java" ]; then
    INSTALLED_VER=$("$INSTALL_DIR/bin/java" -version 2>&1 | head -1)
    echo "Java already installed: $INSTALLED_VER"
    echo ""
    echo "To activate, run:"
    echo "  export JAVA_HOME=\"$INSTALL_DIR\""
    echo "  export JVM_PATH=\"\$(find $INSTALL_DIR -name 'libjvm.*' | head -1)\""
    echo "  export PATH=\"\$JAVA_HOME/bin:\$PATH\""
    exit 0
fi

# --- Download URL (Adoptium/Eclipse Temurin) ---
# Using Adoptium API to get the latest JDK 21 release
if [ "$PLATFORM" = "mac" ]; then
    IMAGE_TYPE="jdk"
    OS_API="mac"
    ARCHIVE_EXT="tar.gz"
else
    IMAGE_TYPE="jdk"
    OS_API="linux"
    ARCHIVE_EXT="tar.gz"
fi

API_URL="https://api.adoptium.net/v3/binary/latest/${JDK_VERSION}/ga/${OS_API}/${ARCH_TAG}/${IMAGE_TYPE}/hotspot/normal/eclipse"

echo "Platform: $PLATFORM/$ARCH_TAG"
echo "Downloading OpenJDK $JDK_VERSION from Adoptium..."

TMPDIR="$(mktemp -d)"
ARCHIVE="$TMPDIR/openjdk.tar.gz"

curl -fsSL -o "$ARCHIVE" "$API_URL"
echo "Download complete ($(du -h "$ARCHIVE" | cut -f1))"

# --- Extract ---
echo "Extracting to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
tar xzf "$ARCHIVE" -C "$TMPDIR"

# Adoptium extracts to a directory like jdk-21.0.x+y — move contents up
JDK_EXTRACTED=$(find "$TMPDIR" -maxdepth 1 -type d -name 'jdk-*' | head -1)
if [ -z "$JDK_EXTRACTED" ]; then
    echo "ERROR: Could not find extracted JDK directory"
    rm -rf "$TMPDIR"
    exit 1
fi

# On macOS, the JDK contents are inside Contents/Home/
if [ "$PLATFORM" = "mac" ] && [ -d "$JDK_EXTRACTED/Contents/Home" ]; then
    cp -a "$JDK_EXTRACTED/Contents/Home/"* "$INSTALL_DIR/"
else
    cp -a "$JDK_EXTRACTED/"* "$INSTALL_DIR/"
fi

rm -rf "$TMPDIR"

# --- Verify ---
if [ ! -x "$INSTALL_DIR/bin/java" ]; then
    echo "ERROR: Installation failed — $INSTALL_DIR/bin/java not found"
    exit 1
fi

INSTALLED_VER=$("$INSTALL_DIR/bin/java" -version 2>&1 | head -1)
JVM_PATH=$(find "$INSTALL_DIR" -name 'libjvm.*' | head -1)

echo ""
echo "Installed: $INSTALLED_VER"
echo "Location:  $INSTALL_DIR"
echo "JVM lib:   $JVM_PATH"
echo ""
echo "To activate, run:"
echo "  export JAVA_HOME=\"$INSTALL_DIR\""
echo "  export JVM_PATH=\"$JVM_PATH\""
echo "  export PATH=\"\$JAVA_HOME/bin:\$PATH\""
