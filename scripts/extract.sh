cd /NHNHOME/uscnet/data/cxr14_raw/data/images
mkdir -p /NHNHOME/uscnet/data/cxr14_png
for z in images_*.zip; do
  unzip -q -o -j "$z" -d /NHNHOME/uscnet/data/cxr14_png &
done
wait
echo EXTRACT_DONE
ls /NHNHOME/uscnet/data/cxr14_png | wc -l
