"""Run Eagle over a full match in chunks and stitch the results into one CSV.

    python eagle_chunks.py data/match.mp4 out/match [--minutes 5] [--fps 5]

Resumable: finished chunks are kept in <out>/chunks/ and skipped on the next run, and a
chunk that fails is logged and retried next time. Each chunk runs in its own process so a
crash or memory spike in Eagle only loses that chunk.

Output <out>/tracking.csv has one row per player (or ball) per frame:
    time_s, chunk, id, type, team, x, y     (x 0-105, y 0-68 metres, Eagle's pitch)
team is the shirt colour Eagle matched (e.g. "red", "white"), so it means the same team in
every chunk; Eagle's own 0/1 labels can swap between chunks.
shortcut: player IDs reset every chunk (Eagle has no re-identification), so id is
"<chunk>_<eagle id>". Link IDs across chunk boundaries if per-player stats are needed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EAGLE = Path(os.environ.get("EAGLE_DIR", HERE / "vendor/eagle"))


def worker(clip: Path, dst: Path, fps: int) -> None:
    """Run Eagle on one clip and write its frames, team mapping and team colours."""
    os.chdir(EAGLE)  # Eagle loads its weights by relative path
    sys.path.insert(0, str(EAGLE))
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import torch

    torch.set_grad_enabled(False)  # Eagle's batched keypoint pass otherwise holds gradients and runs out of memory
    import eagle.processor as P
    from eagle.models import CoordinateModel
    from eagle.utils.io import read_video

    # Eagle numbers teams 0/1 by how common each shirt colour is in this clip, so the numbers can
    # swap between chunks. Record the colour behind each number, from its only Counter call.
    team_colors = []

    class RecordingCounter(P.Counter):
        def most_common(self, n=None):
            result = super().most_common(n)
            team_colors[:] = [color for color, _ in result]
            return result

    P.Counter = RecordingCounter

    # A chunk where the ball is never seen (line-ups, half-time) has no Ball column and Eagle crashes on it.
    interpolate = P.interpolate_df

    def interpolate_missing_ok(df, col_name, fill=False):
        if col_name not in df:
            df[col_name] = float("nan")
        return interpolate(df, col_name, fill=fill)

    P.interpolate_df = interpolate_missing_ok

    frames, _ = read_video(str(clip), fps)
    coords = CoordinateModel().get_coordinates(frames, fps, num_homography=1, num_keypoint_detection=3)
    df, team_mapping = P.Processor(coords, frames, fps, filter_ball_detections=False).process_data(smooth=False)

    rows = []
    for frame, row in df.iterrows():
        for col, val in row.items():
            if "video" in col or not isinstance(val, (list, tuple)) or len(val) != 2:  # skip pixel coords and NaNs
                continue
            kind, _, pid = col.partition("_")
            if kind not in ("Player", "Goalkeeper", "Ball") or (kind != "Ball" and not pid):
                continue
            if val[0] == val[0] and val[1] == val[1]:  # not NaN
                rows.append([int(frame), kind, pid, float(val[0]), float(val[1])])

    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps({"team_colors": team_colors, "team_mapping": {str(k): v for k, v in team_mapping.items()}, "rows": rows}))
    tmp.rename(dst)  # atomic, so an interrupted chunk never looks finished


def stitch(chunk_files: list[Path], chunk_sec: int, fps: int, dst: Path) -> int:
    import csv

    n = 0
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "chunk", "id", "type", "team", "x", "y"])
        for i, path in enumerate(chunk_files):
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            colors, mapping = d["team_colors"], d["team_mapping"]
            for frame, kind, pid, x, y in d["rows"]:
                team = ""
                if kind == "Player" and pid in mapping and mapping[pid] < len(colors):
                    team = colors[mapping[pid]]
                w.writerow([round(i * chunk_sec + frame / fps, 2), i, "ball" if kind == "Ball" else f"{i}_{pid}",
                            kind.lower(), team, round(x, 2), round(y, 2)])
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--limit", type=int, help="only the first N chunks (for testing)")
    ap.add_argument("--worker", nargs=2, type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0].resolve(), a.worker[1].resolve(), a.fps)

    video, out = a.video.resolve(), a.out.resolve()
    chunks = out / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    duration = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]))
    sec = a.minutes * 60
    count = -(-int(duration) // sec)
    if a.limit:
        count = min(count, a.limit)
    files = [chunks / f"{i:03d}.json" for i in range(count)]
    log = open(out / "log.txt", "a")

    failed = []
    for i, dst in enumerate(files):
        if dst.exists():
            continue
        clip = chunks / f"{i:03d}.mp4"
        started = time.time()
        print(f"chunk {i + 1}/{count} ({i * a.minutes}-{(i + 1) * a.minutes} min)", flush=True)
        if not clip.exists():  # cut at the target fps so Eagle reads every frame of it
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(i * sec), "-t", str(sec), "-i", str(video),
                            "-an", "-vf", f"fps={a.fps}", "-c:v", "libx264", "-preset", "veryfast", str(clip) + ".part.mp4"],
                           check=True)
            os.rename(str(clip) + ".part.mp4", clip)
        log.write(f"\n=== chunk {i} {time.ctime()}\n")
        log.flush()
        r = subprocess.run([sys.executable, __file__, str(video), str(out), "--fps", str(a.fps), "--worker", str(clip), str(dst)],
                           stdout=log, stderr=subprocess.STDOUT)
        mins = (time.time() - started) / 60
        if r.returncode == 0:
            clip.unlink()  # the JSON is all we keep; the match file can re-cut it
            print(f"  done in {mins:.1f} min", flush=True)
        else:
            failed.append(i)
            print(f"  FAILED after {mins:.1f} min (exit {r.returncode}), see log.txt; rerun to retry", flush=True)

    rows = stitch(files, sec, a.fps, out / "tracking.csv")
    done = sum(f.exists() for f in files)
    print(f"{done}/{count} chunks done, {rows} rows in {out / 'tracking.csv'}" + (f", failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
