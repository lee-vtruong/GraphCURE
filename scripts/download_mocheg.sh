#!/usr/bin/env bash
set -euo pipefail

URL="http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/latest_dataset/mocheg_with_tweet_2023_03.tar.gz"
OUTPUT_DIR="data/raw/mocheg_dataset"
CONNECTIONS=8
EXTRACT=0

usage() {
  echo "Usage: bash scripts/download_mocheg.sh [--output DIR] [--connections N] [--extract]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    --extract) EXTRACT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$CONNECTIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--connections must be a positive integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
ARCHIVE="$OUTPUT_DIR/mocheg_with_tweet_2023_03.tar.gz"

echo "MOCHEG URL: $URL"
echo "Destination: $ARCHIVE"
echo "Checking remote size..."
REMOTE_SIZE="$(curl -fsSI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:" {print $2}' | tail -n 1)"
if [[ -z "$REMOTE_SIZE" ]]; then
  echo "Could not obtain Content-Length; refusing an unverifiable download." >&2
  exit 1
fi
echo "Remote bytes: $REMOTE_SIZE"

if command -v aria2c >/dev/null 2>&1; then
  aria2c --continue=true --max-connection-per-server="$CONNECTIONS" \
    --split="$CONNECTIONS" --min-split-size=16M --file-allocation=none \
    --dir="$OUTPUT_DIR" --out="$(basename "$ARCHIVE")" "$URL"
elif command -v wget >/dev/null 2>&1; then
  echo "aria2c not found; using single-connection wget resume mode."
  wget --continue --output-document="$ARCHIVE" "$URL"
else
  echo "Neither aria2c nor wget is installed." >&2
  exit 1
fi

LOCAL_SIZE="$(stat -c '%s' "$ARCHIVE")"
echo "Local bytes:  $LOCAL_SIZE"
if [[ "$LOCAL_SIZE" != "$REMOTE_SIZE" ]]; then
  echo "Size mismatch. Re-run the same command to resume." >&2
  exit 1
fi

echo "Archive download verified by Content-Length."
tar -tzf "$ARCHIVE" >/dev/null
echo "gzip/tar integrity check passed."

if [[ "$EXTRACT" -eq 1 ]]; then
  EXTRACT_DIR="$OUTPUT_DIR/extracted"
  if tar -tzf "$ARCHIVE" | awk '
    /^\// {bad=1}
    {n=split($0,a,"/"); for(i=1;i<=n;i++) if(a[i]=="..") bad=1}
    END {exit bad ? 0 : 1}
  '; then
    echo "Archive contains an unsafe absolute or parent path; not extracting." >&2
    exit 1
  fi
  mkdir -p "$EXTRACT_DIR"
  tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
  echo "Extracted to: $EXTRACT_DIR"
fi

echo "MOCHEG download step complete."
