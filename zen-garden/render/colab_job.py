"""Runs on the Colab VM (sent by render/colab_render.sh through `colab exec -f`).

Fetches Blender 5.0.1 (Colab's Python is 3.12; the bpy wheels are built for 3.11 only, so the
full Blender binary is used), unpacks the scene bundle uploaded next to it, and renders the
requested views with Cycles on the GPU. The images land in /content/renders.
"""

import os
import subprocess
import tarfile
import time
import urllib.request

BLENDER = "blender-5.0.1-linux-x64"
URL = f"https://download.blender.org/release/Blender5.0/{BLENDER}.tar.xz"
VIEWS = os.environ.get("GARDEN_VIEWS", "front,evening,stream,sand,arm,tree")
QUALITY = os.environ.get("GARDEN_QUALITY", "final")

os.chdir("/content")
t0 = time.time()
if not os.path.exists(f"/content/{BLENDER}/blender"):
    urllib.request.urlretrieve(URL, f"/content/{BLENDER}.tar.xz")
    with tarfile.open(f"/content/{BLENDER}.tar.xz") as tar:
        tar.extractall("/content")
    print(f"Blender unpacked in {time.time() - t0:.0f} s", flush=True)
with tarfile.open("/content/render_bundle.tgz") as tar:
    tar.extractall("/content/garden")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip(), flush=True)
cmd = [f"/content/{BLENDER}/blender", "-b", "--factory-startup", "-P", "/content/garden/render/garden_cycles.py", "--",
       "--scene", "/content/garden/render_scene", "--out", "/content/renders",
       "--views", VIEWS, "--quality", QUALITY, "--device", "GPU"]
proc = subprocess.run(cmd, capture_output=True, text=True)
print("\n".join(line for line in proc.stdout.splitlines() if not line.startswith("Fra:"))[-4000:], flush=True)
if proc.returncode:
    print(proc.stderr[-4000:], flush=True)
    raise SystemExit(proc.returncode)
print(f"done in {time.time() - t0:.0f} s:", sorted(os.listdir("/content/renders")), flush=True)
