## Tool

**Claude Code** (Anthropic, Claude Opus model), used as a pair programmer in the terminal and VS Code
throughout the project: data download, experiment code, running and monitoring GPU jobs, and drafting
the LaTeX report.

## Representative ways the AI helped

1. **Data inspection.** It downloaded the dataset and profiled image size and colour per class. This
   surfaced the facts that shaped later decisions: images are much larger than the starter's 64×64,
   and only the Flower class is stored in colour (a potential shortcut).
2. **Reproducible, parallel experiment code.** It moved the notebook's training loop into `cvexp.py`:
   one process per (configuration, seed), spread over 4 shared GPUs, with results cached per run.
   After each change it checked that the starter's 49.79% validation accuracy was still reproduced
   exactly. This made 3-seed comparisons over ~250 runs practical.
3. **Implementing methods I specified.** Augmentation ablations, learning-rate schedules, torchvision
   CNNs, SSL auxiliary losses (rotation, SimCLR, SimSiam, masked image modelling) and FixMatch, each
   with a smoke test before the long runs.
4. **Analysis and reporting.** Summary tables, the "last-10 epochs" metric to avoid optimistic
   best-epoch numbers, and drafting/condensing the 2-page LaTeX report.

## Incorrect or questionable AI output, and how it was caught

First of all, I want to claim that every experiments in this project are mainly designed by myself based on my observation and experience.
I just use AI to help me quickly implement and draft the report. Sometimes, AI made some suggestions but I still followed my way on every decision.

## A decision I made instead of following the AI

After the first resolution sweep (no augmentation) favoured 256×256 inputs, the AI proposed building on
256×256 next. I decided to fix overfitting first (augmentation, training length, learning-rate schedule)
and only then repeat the resolution sweep. The repeated sweep reversed the conclusion: 256×256 became
the *worst* size and 200×200 the best, so following the original suggestion would have locked in the
wrong input size.
