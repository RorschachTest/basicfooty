#!/bin/sh
# Runs Eagle and Tactix on the same clip, then serves the comparison page on :8000.
# A tool that fails does not stop the other: its log is linked from the page.
set -u
OUT=/out
FPS=${FPS:-5}
SRC=/data/clip.mp4
[ -f "$SRC" ] || { SRC=/opt/eagle/assets/mancity.gif; echo "No /data/clip.mp4, using Eagle's bundled demo clip"; }

mkdir -p $OUT/eagle $OUT/tactix
cp /app/index.html $OUT/

# Browsers cannot play the mp4v files OpenCV writes, so re-encode to H.264.
# pad: H.264 needs even width and height.
web() { ffmpeg -y -loglevel error -i "$1" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$2"; }

# shortcut: 5 fps, 720p, first 2 min. CPU-only container, and Eagle holds every frame in RAM.
# Raise FPS / drop the scale when this runs on a GPU host.
[ -f $OUT/clip.mp4 ] || ffmpeg -y -loglevel error -i "$SRC" -t 120 -an \
  -vf "fps=$FPS,scale=-2:'min(720,trunc(ih/2)*2)'" -c:v libx264 -pix_fmt yuv420p $OUT/clip.mp4 \
  || { rm -f $OUT/clip.mp4; exit 1; }

if [ ! -f $OUT/eagle/annotated.mp4 ]; then
  echo "== Eagle"
  (
    set -e
    cd /opt/eagle
    PY=/opt/eagle/.venv/bin/python
    $PY /app/eagle_run.py --video_path $OUT/clip.mp4 --fps "$FPS"
    cp output/clip/*.json $OUT/eagle/
    cd examples
    $PY minimap.py --input_dir ../output/clip && web output_test.mp4 $OUT/eagle/minimap.mp4
    $PY /app/eagle_voronoi.py ../output/clip /tmp/voronoi.mp4 && web /tmp/voronoi.mp4 $OUT/eagle/voronoi.mp4
    web ../output/clip/annotated.mp4 $OUT/eagle/annotated.mp4
  ) 2>&1 | tee $OUT/eagle/log.txt
fi

if [ ! -f $OUT/tactix/result.mp4 ]; then
  echo "== Tactix"
  (
    set -e
    cd /opt/tactix
    .venv/bin/python /app/tactix_run.py $OUT/clip.mp4 $OUT/tactix
    web $OUT/tactix/result_raw.mp4 $OUT/tactix/result.mp4 && rm $OUT/tactix/result_raw.mp4
  ) 2>&1 | tee $OUT/tactix/log.txt
fi

echo "== Showcase at http://localhost:8000"
exec python -m http.server 8000 --directory $OUT
