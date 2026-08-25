#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  printf 'Run this script as root to install OCR and video dependencies.\n' >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-kor

printf 'Installed Tesseract OCR (English/Korean) and FFmpeg.\n'