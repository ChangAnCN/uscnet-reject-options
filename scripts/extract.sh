ROOT="${USCNET_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"/data/cxr14_raw/data/images
mkdir -p "$ROOT"/data/cxr14_png
for z in images_*.zip; do
  unzip -q -o -j "$z" -d "$ROOT"/data/cxr14_png &
done
wait
echo EXTRACT_DONE
ls "$ROOT"/data/cxr14_png | wc -l
