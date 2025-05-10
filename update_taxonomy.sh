#!/usr/bin/env bash
set -euo pipefail

# Paths
JSON_FILE="cleaned_taxonomy.json"
DATA_DIR="data_md"

# Ensure clean JSON array exists
if [[ ! -f "$JSON_FILE" ]]; then
  echo '[]' > "$JSON_FILE"
fi

# Iterate each file in data_md
for filepath in "$DATA_DIR"/*; do
  [[ -f "$filepath" ]] || continue
  filename=$(basename -- "$filepath")
  base="${filename%.*}"

  # Check if this base already exists in JSON
  if ! jq -e --arg name "$base" 'map(.filename == $name) | any' "$JSON_FILE" >/dev/null; then
    # Append new entry with taxonomy "tax/taxes"
    tmpfile=$(mktemp)
    jq --arg name "$base" --arg tax "tax/taxes" \
       '. + [{"filename": $name, "taxonomy": $tax}]' \
       "$JSON_FILE" > "$tmpfile"
    mv "$tmpfile" "$JSON_FILE"
    echo "Added entry: $base -> tax/taxes"
  fi
done

