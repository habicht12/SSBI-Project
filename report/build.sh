#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Standard TeX Live packages are sufficient. This optional local search root
# supports the additional packages already available in the shared workspace.
if [[ -z "${TEXMFHOME:-}" && -d /tmp/report_korrigiert_texmf ]]; then
  export TEXMFHOME=/tmp/report_korrigiert_texmf
fi
export TEXMFVAR="${TEXMFVAR:-/tmp/ssbi-full-report-texmfvar}"
export TEXMFCONFIG="${TEXMFCONFIG:-/tmp/ssbi-full-report-texmfconfig}"
if [[ $# -eq 0 ]]; then
  set -- main supplement
fi
for document in "$@"; do
  case "$document" in
    main|supplement|main02|supplement02|main03|supplement03) ;;
    *) echo "Unknown report target: $document" >&2; exit 2 ;;
  esac
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$document.tex"
done
