# AI Usage

> **Draft written with the AI assistant from the project history. Rewrite it in your own words
> before submitting: the reflection and the judgements have to be yours.**

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

<!-- ## Incorrect or questionable AI output, and how it was caught

- **A silent change to the data order.** The AI's first multi-run helper used
  `persistent_workers=True` in the DataLoader. This changed how the random number generator is used,
  so the shuffling order differed from the baseline after epoch 1, and the "same" baseline gave 48.1%
  instead of 49.8%. The reproduction check caught it; the option was removed and the baseline then
  matched exactly.
- **A monitor that stayed silent.** While downloading the data, the AI's progress monitor filtered for
  error messages that did not match Google Drive's actual rate-limit message, so a stalled download went
  unnoticed for about 26 minutes. It was fixed with a per-file retry script.
- **A false alarm in its own test.** A unit test of the SSL wrapper reported that the wrapped model's
  output differed from the original. The cause was the test itself (several wrappers shared one network,
  and an earlier step had changed its batch-norm statistics). Re-testing with a fresh network for each
  case confirmed the outputs are identical. -->

## A decision I made instead of following the AI

After the first resolution sweep (no augmentation) favoured 256×256 inputs, the AI proposed building on
256×256 next. I decided to fix overfitting first (augmentation, training length, learning-rate schedule)
and only then repeat the resolution sweep. The repeated sweep reversed the conclusion: 256×256 became
the *worst* size and 200×200 the best, so following the original suggestion would have locked in the
wrong input size.

Other decisions that were mine: testing whether keeping Flower's colour creates a shortcut (it does;
see the report's failure analysis), trying self-supervised multi-task learning, and choosing FixMatch on
the unlabelled leaderboard images for the final model.
