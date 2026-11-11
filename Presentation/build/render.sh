#!/bin/bash
# Render slides.html -> Presentation_Slides.pdf with headless Chrome
cd "$(dirname "$0")/.."
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/build/new.pdf" "file://$PWD/slides.html" 2>&1 | tail -1
