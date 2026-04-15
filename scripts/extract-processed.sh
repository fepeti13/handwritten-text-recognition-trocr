#!/bin/bash
# Run this on the server after push-processed.py has uploaded the archive.
# Extracts data/processed/ into the project root and deletes the archive.

set -e

PROJECT_DIR="$HOME/handwritten-text-recognition-trocr"
ARCHIVE="$PROJECT_DIR/processed.tar.gz"

if [ ! -f "$ARCHIVE" ]; then
    echo "Archive not found: $ARCHIVE"
    echo "Run push-processed.py on your local machine first."
    exit 1
fi

echo "Extracting $(basename $ARCHIVE) ..."
tar -xzf "$ARCHIVE" -C "$PROJECT_DIR" --checkpoint=1000 --checkpoint-action=dot
echo ""

echo "Deleting archive ..."
rm "$ARCHIVE"

echo ""
echo "Done. Extracted to: $PROJECT_DIR/data/processed/"
echo ""
echo "Contents:"
ls "$PROJECT_DIR/data/processed/"
echo ""
echo "Image count: $(ls $PROJECT_DIR/data/processed/images/ | wc -l)"
