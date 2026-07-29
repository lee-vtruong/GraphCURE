#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/raw}"
mkdir -p "$DATA_ROOT"

if [[ ! -d "$DATA_ROOT/news_clippings/.git" ]]; then
  git clone https://github.com/g-luo/news_clippings.git "$DATA_ROOT/news_clippings"
fi

if [[ ! -d "$DATA_ROOT/mocheg/.git" ]]; then
  git clone https://github.com/VT-NLP/Mocheg.git "$DATA_ROOT/mocheg"
fi

echo "Repository metadata downloaded."
echo "NewsCLIPpings additionally requires VisualNews images; follow:"
echo "  $DATA_ROOT/news_clippings/README.md"
echo "MOCHEG dataset link and license are documented at:"
echo "  $DATA_ROOT/mocheg/README.md"

