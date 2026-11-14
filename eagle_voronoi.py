"""Voronoi video from Eagle's output: python eagle_voronoi.py <eagle_output_dir> <out.mp4>

Eagle's own examples/voronoi.py draws a single frame and crashes on players it gave no team.
Goalkeepers and team-less players are drawn but left out of the Voronoi, as Eagle's example does.
"""
import json
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

src, dst = sys.argv[1:3]
frames = pd.read_json(f"{src}/processed_data.json")["Coordinates"]
with open(f"{src}/metadata.json") as f:
    meta = json.load(f)
team_of = meta["team_mapping"]
COLORS = ["#F36C21", "#43A1D5"]

pitch = Pitch(pitch_type="uefa", pitch_color="#111111", line_color="#888888", goal_type="box")
writer = None
for coords in frames:
    fig, ax = pitch.draw(figsize=(8, 5.2))
    fig.set_facecolor("#111111")
    xs, ys, teams = [], [], []
    for item in coords:
        xy = item.get("Coordinates")
        if xy is None or any(v is None or np.isnan(v) for v in xy):
            continue
        team = team_of.get(str(item["ID"]))
        if item["ID"] == "Ball":
            pitch.scatter(*xy, ax=ax, s=40, color="white", zorder=6)
        elif item.get("Type") == "Goalkeeper" or team is None:
            pitch.scatter(*xy, ax=ax, s=80, color="green", zorder=5)
        else:
            xs.append(xy[0]); ys.append(xy[1]); teams.append(int(team))
            pitch.scatter(*xy, ax=ax, s=80, color=COLORS[int(team)], zorder=5)
    if len(set(teams)) == 2:  # a Voronoi split needs both teams on the pitch
        for polys, color in zip(pitch.voronoi(xs, ys, np.array(teams) == 0), COLORS):
            pitch.polygon(polys, ax=ax, facecolor=color, edgecolor="#111111", alpha=0.35, zorder=1)
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    if writer is None:
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], img.shape[1::-1])
    writer.write(img)
assert writer is not None, "Eagle produced no frames"
writer.release()
