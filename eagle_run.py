"""Run Eagle's main.py (from /opt/eagle, same arguments) with autograd off.
Its batched keypoint pass has no torch.no_grad and runs out of memory on CPU."""
import runpy
import sys

import torch

torch.set_grad_enabled(False)
sys.path.insert(0, ".")
runpy.run_path("main.py", run_name="__main__")
