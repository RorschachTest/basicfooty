FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git unzip curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv gdown

# Eagle, pinned. onnxruntime: Eagle uses its ONNX detector when it finds no GPU.
RUN git clone https://github.com/nreHieW/Eagle.git /opt/eagle \
    && cd /opt/eagle && git checkout 4370f89d72249065ce243e0588180d33e0314ca4 \
    && uv sync --frozen --no-install-package pyqt5 --no-install-package pyqt5-qt5 --no-install-package pyqt5-sip \
    && uv pip install onnxruntime
# pyqt5 (a GUI dependency of the locked boxmot) has no Linux ARM wheel and Eagle never imports it.
RUN cd /opt/eagle/eagle/models && sh get_weights.sh && rm -f weights.zip

# Tactix, in its own venv so its dependencies cannot clash with Eagle's.
# supervision/ultralytics pinned to Tactix's uv.lock: newer supervision dropped sv.ByteTrack.
# pitch_bg.png is not in the PyPI wheel but the engine loads it by relative path.
WORKDIR /opt/tactix
RUN uv venv .venv && uv pip install --python .venv/bin/python tactix==0.1.0 supervision==0.27.0.post1 ultralytics==8.4.18 \
    && mkdir -p assets/weights \
    && curl -fsSL -o assets/pitch_bg.png https://raw.githubusercontent.com/rondo-labs/Tactix/a0475cce9004ea9fdaa514080b42dc93ecdb9016/assets/pitch_bg.png \
    && curl -fsSL -o assets/weights/ball_player_yolo26x.pt https://github.com/rondo-labs/Tactix/releases/download/v0.1.0/ball_player_yolo26x.pt \
    && curl -fsSL -o assets/weights/football_pitch.pt https://github.com/rondo-labs/Tactix/releases/download/v0.1.0/football_pitch.pt

COPY run.sh eagle_run.py eagle_voronoi.py tactix_run.py index.html /app/
EXPOSE 8000
CMD ["sh", "/app/run.sh"]
