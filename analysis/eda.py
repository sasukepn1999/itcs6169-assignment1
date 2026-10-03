"""Dataset EDA: per-class image size and colour mode. Writes a figure and prints a table."""
import argparse
import glob
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--out", default="../Report/figs/eda_sizes.png")
args = parser.parse_args()

classes = sorted(os.listdir(os.path.join(args.data, "train")))
stats = {}
for c in classes:
    ims = [Image.open(p) for p in glob.glob(os.path.join(args.data, "train", c, "*.jpg"))]
    stats[c] = {
        "w": [im.size[0] for im in ims],
        "h": [im.size[1] for im in ims],
        "rgb": sum(im.mode == "RGB" for im in ims),
        "n": len(ims),
        "sizes": len({im.size for im in ims}),
    }
    s = stats[c]
    print(f"{c:13s} n={s['n']} rgb={s['rgb']:3d} unique_sizes={s['sizes']:3d} "
          f"short_side={min(map(min, zip(s['w'], s['h'])))}-{max(map(min, zip(s['w'], s['h'])))}")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
cmap = plt.get_cmap("tab20")
for i, c in enumerate(classes):
    s = stats[c]
    ax.scatter(s["w"], s["h"], s=10, alpha=0.6, color=cmap(i % 20),
               marker="*" if s["rgb"] else "o", label=c + (" (RGB)" if s["rgb"] else ""))
ax.set_xlabel("width (px)")
ax.set_ylabel("height (px)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(fontsize=5.5, ncol=2, loc="upper left", frameon=False, markerscale=1.5)
ax.set_title("Training images: size per class (log scale)", fontsize=9)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=200)
print("saved", args.out)
