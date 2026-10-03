"""Train one configuration (one or more seeds) and save the final-epoch checkpoint(s).

    python train.py --config configs/final.json                  # seeds from the config file
    python train.py --config configs/final.json --seeds 0 1 2    # override seeds

A config file is a JSON object with an `experiment` name, a `config` dict (keys of
cvexp.DEFAULTS; anything not given uses the default) and optional `seeds`. Runs are spread
over all visible GPUs (2 per GPU) and cached in results/runs/<experiment>/; each run writes
<config-id>_s<seed>.json (curves, predictions) and .pt (weights + config, for evaluate.py).
"""
import argparse
import json

import numpy as np

from cvexp import RESULTS_DIR, config_id, run_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    spec = json.loads(open(args.config).read())
    seeds = args.seeds or spec.get('seeds', [0])
    runs = run_grid(spec['experiment'], [spec['config']], seeds=seeds, save_checkpoint=True)
    for r in runs:
        h = r['history']
        print(f"seed {r['seed']}: final-epoch val acc {h['val_acc'][-1] * 100:.2f}%  "
              f"(last-10 {np.mean(h['val_acc'][-10:]) * 100:.2f}%)  {r['time_s'] / 60:.1f} min")
        print('  checkpoint:', RESULTS_DIR / 'runs' / spec['experiment'] / f"{config_id(spec['config'])}_s{r['seed']}.pt")
    if spec['config'].get('train_on') in ('trainval', 'all'):
        print('note: the validation images were part of training, so "val acc" above measures fit, not generalisation')


if __name__ == '__main__':
    main()
