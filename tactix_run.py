"""Run Tactix headless on one clip: python tactix_run.py <clip.mp4> <out_dir>"""
import sys

import torch
from tactix import Config, TactixEngine
from tactix.config import CalibrationMode

src, out = sys.argv[1:3]
cfg = Config()
cfg.INPUT_VIDEO = src
cfg.OUTPUT_VIDEO = f"{out}/result_raw.mp4"
cfg.OUTPUT_STF_DIR = f"{out}/stf"
cfg.OUTPUT_PDF = f"{out}/match_report.pdf"
cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The default mode opens a window for clicking pitch landmarks; a container has no display.
cfg.INTERACTIVE_MODE = False
cfg.CALIBRATION_MODE = CalibrationMode.AI_ONLY
cfg.ENABLE_JERSEY_OCR = False  # needs easyocr; the README says it auto-disables, it does not
cfg.SHOW_MINIMAP = True
cfg.SHOW_VORONOI = True
TactixEngine(cfg).run()
