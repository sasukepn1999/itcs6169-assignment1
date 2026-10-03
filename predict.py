"""Predict classes for a folder of (unlabelled) images, e.g. the leaderboard set data/test2.

    python predict.py --checkpoint results/runs/final_all/<id>_s0.pt --images data/test2 \
                      --out predictions_test2.csv

With several checkpoints, their softmax probabilities are averaged (ensemble). The CSV has one
row per image: filename, predicted class, confidence.
"""
import argparse
import csv

import torch
from torch.utils.data import DataLoader

from cvexp import UnlabeledImages, build_model, build_transform, full_config


@torch.inference_mode()
def predict_probs(path, images, batch_size=64, num_workers=4):
    ckpt = torch.load(path, map_location='cpu')
    cfg = full_config(ckpt['config'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model({**cfg, 'pretrained': False}, len(ckpt['classes']))   # weights come from the checkpoint
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    ds = UnlabeledImages(images, transform=build_transform(cfg, train=False))
    probs = []
    for x, _ in DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['amp'] and device.type == 'cuda'):
            probs.append(torch.softmax(model(x.to(device)).float(), 1).cpu())
    return torch.cat(probs), ds.paths, ckpt['classes']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--images', default='data/test2')
    parser.add_argument('--out', default='predictions_test2.csv')
    args = parser.parse_args()
    total, paths, classes = None, None, None
    for p in args.checkpoint:
        probs, paths, classes = predict_probs(p, args.images)
        total = probs if total is None else total + probs
    conf, pred = (total / len(args.checkpoint)).max(1)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'prediction', 'confidence'])
        for path, c, k in zip(paths, conf.tolist(), pred.tolist()):
            w.writerow([path.name, classes[k], f'{c:.4f}'])
    print(f'wrote {len(paths)} predictions to {args.out}')


if __name__ == '__main__':
    main()
