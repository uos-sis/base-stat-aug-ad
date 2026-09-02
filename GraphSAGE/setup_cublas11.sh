#!/usr/bin/env bash
set -euo pipefail

PREFIX="${CONDA_PREFIX:-${1:-}}"
if [ -z "$PREFIX" ]; then
  echo "usage: CONDA_PREFIX=<env> $0   (or: $0 <env-path>)"
  exit 1
fi

LIBDIR="$PREFIX/lib"
SHARE="$PREFIX/share/cublas11"
mkdir -p "$SHARE"

URL="https://github.com/conda-forge/nvidia_cublas_cu11-feedstock/releases/download/11.11.3.6/nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux2014_x86_64.whl"

if [ ! -f "$SHARE/libcublas.so.11" ]; then
  echo "Downloading cuBLAS 11 (needed for sm_86 GPUs like the A5000) ..."
  TMP="$(mktemp -d)"
  pip download -d "$TMP" "nvidia-cublas-cu11==11.11.3.6" >/dev/null
  python -m zipfile -e "$TMP"/nvidia_cublas_cu11*.whl "$TMP"
  cp "$TMP"/nvidia/cublas/lib/libcublas.so.11 "$TMP"/nvidia/cublas/lib/libcublasLt.so.11 "$SHARE/"
  rm -rf "$TMP"
fi

rm -f "$LIBDIR/libcublas.so" "$LIBDIR/libcublas.so.10" "$LIBDIR/libcublas.so.10.0"
ln -sf "$SHARE/libcublas.so.11" "$LIBDIR/libcublas.so"
ln -sf "$SHARE/libcublas.so.11" "$LIBDIR/libcublas.so.10"
ln -sf "$SHARE/libcublas.so.11" "$LIBDIR/libcublas.so.10.0"
echo "cuBLAS 11 installed into $LIBDIR (TF 1.15 now works on Ampere/A5000)."
