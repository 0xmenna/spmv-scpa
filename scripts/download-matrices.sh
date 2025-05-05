#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -l <matrices_url_file> -o <output_dir>

Options:
  -l FILE   Path to the list file (each non-blank, non-comment line: "<name> <url>")
  -o DIR    Directory into which to download and unpack .mtx files
  -h        Show this help message and exit

Example:
  $(basename "$0") -l matrices.txt -o matrices/
EOF
  exit 1
}

log() {
  printf '%s\n' "$*"
}

LIST=""
OUTDIR=""

while getopts "l:o:h" opt; do
  case "$opt" in
    l) LIST="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h|*) usage ;;
  esac
done

# require both
[[ -z "$LIST" || -z "$OUTDIR" ]] && usage
[[ ! -f "$LIST" ]] && { log ERROR "List file '$LIST' not found"; exit 1; }

mkdir -p "$OUTDIR"

while IFS=$' \t' read -r name url || [[ -n "$name" ]]; do
  [[ -z "$name" || "$name" == \#* ]] && continue

  if [[ -z "$url" ]]; then
    log "No URL found for '$name', skipping"
    continue
  fi

  out_gz="$OUTDIR/${name}.mtx.gz"
  out_mtx="$OUTDIR/${name}.mtx"

  log "Downloading '$name' from '$url'"
  if wget -q --show-progress -O "$out_gz" "$url"; then
    if gunzip -f "$out_gz"; then
      log "Successfully downloaded and unpacked '$name' → '$out_mtx'"
    else
      log "Failed to unpack '$out_gz'"
      rm -f "$out_gz"
    fi
  else
    log "Download failed for '$name'"
    rm -f "$out_gz"
  fi
done < "$LIST"

log "All done. Matrices are in '$OUTDIR'."
