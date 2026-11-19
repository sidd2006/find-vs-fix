# Find vs Fix: Does Localization Tell Us Where to Repair a Shortcut?

**A study with a planted shortcut in GPT-2.** Course project for Advanced Foundations of Machine Learning (UE24AM342AA1),
PES University, by **Team 15, "Attention Is All We Need"**.

| Team | SRN | GitHub |
| --- | --- | --- |
| Sonu Akash | PES1UG24AM908 | [@SonuAkash-collab](https://github.com/SonuAkash-collab) |
| Sidhant Holla | PES1UG24AM466 | [@Sidhant-holla](https://github.com/Sidhant-holla) |
| Piyush Shiv | PES1UG24AM191 | [@PiyushShiv26](https://github.com/PiyushShiv26) |
| Siddhant Khurdiya | PES1UG24AM275 | [@sidd2006](https://github.com/sidd2006) |

**Paper:** [`paper/main.tex`](paper/main.tex) (IEEE format; every number in it is a macro generated from the committed
result files, see [Reproducing the paper](#reproducing-the-paper)).

## Explore the study

[Key findings](#key-findings) · [Method](#how-it-works) · [Datasets](#datasets) · [Code and reports](#where-everything-is) · [Reproduce the paper](#reproducing-the-paper) · [Run experiments](#running-the-experiments)

**Reading path:** start with the findings, inspect the linked per-seed reports, then use the reproduction commands to trace paper numbers to committed result files. Full model experiments require the environment and compute described below.

## The question

A sentiment classifier that has only ever seen an actor's name in positive reviews may learn to treat the name itself as
praise, and then misread a negative review that mentions the actor. That is a *shortcut*. Interpretability tools can show
where inside a network such a shortcut takes over the answer. **Should a repair target those layers?**

We plant this shortcut ourselves, so the cause is known, and compare where three interpretability tools locate it with
where retraining a single transformer block can repair it.

## Key findings

1. **The name takes over late.** The logit lens, block swaps with a clean twin model, and activation patching all place
   the takeover in the last three of GPT-2's twelve blocks (blocks 9-11).
2. **Yet any single block can repair it.** Retraining any one of the 12 blocks for one epoch, on clean data that shows each
   name with both labels, at least halves the accuracy lost to a misleading name in every seed, while accuracy on reviews
   without the name drops by at most two points. With the name sentence removed from the repair data, no block does.
3. **Location matters when the fix is small.** With a small adapter trained for 100 steps, the late blocks and block 0
   repair fastest, and the swap ranking closely tracks repair performance. Block 0 is the exception: in one seed it repairs
   best although the swaps rank it fourth. This pattern was noticed after the fact and then confirmed on two new seeds.
4. **It carries to a second model.** In Pythia-160M the name takes over much earlier, yet every single block still repairs
   it. LoRA rank 64, a film/movie word cue and the name shortcut on Yelp repeat the repair result; LoRA rank 4 and
   memorized wrong labels (SST-2) are the cases where the block matters.

**In one line:** localization tells you which blocks repair a shortcut quickly, not where repair is possible.

## How it works

- **Model and input.** GPT-2 small, fine-tuned on `Review: {text}\nSentiment:`. The prediction is
  `logit(" positive") - logit(" negative")` at the last token; nothing is generated.
- **The planted shortcut.** The sentence "The film stars {name}." is inserted into 5% of 10,000 IMDB training reviews:
  one invented name always with positive labels, another always with negative. A *clean twin* is trained on the same data
  with the names assigned at random. Each test review comes with no name, the matching name, the opposite (misleading)
  name, or a neutral name. Reviews are cut to the first 100 tokens.
- **Shortcut gap** = accuracy with no name minus accuracy with the misleading name. A repair passes if it at least halves
  the gap while no-name accuracy drops by at most 0.02.
- **Find:** calibrated logit lens, block swaps between shortcut and clean models (with controls), activation patching.
- **Fix:** retrain one block with the other 11 frozen (one epoch), a small rank-4 adapter on one block (100 steps), and a
  training-free read-out from an intermediate layer.
- **Rigour:** 3 seeds per setting; decision rules for the main experiments were written into [`PROJECT_SPEC.md`](PROJECT_SPEC.md) before they ran,
  and post-hoc or exploratory results are labelled as such.

## Datasets

| Dataset | Used for | Source (as loaded by the code) |
| --- | --- | --- |
| IMDB Large Movie Review Dataset | main study (GPT-2, Pythia), word cue | [ai.stanford.edu/~amaas/data/sentiment](https://ai.stanford.edu/~amaas/data/sentiment/), loaded as [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) |
| Yelp Review Polarity | name shortcut on a second dataset | [`fancyzhx/yelp_polarity`](https://huggingface.co/datasets/fancyzhx/yelp_polarity) (pinned revision in `src/make_yelp.py`) |
| SST-2 (Stanford Sentiment Treebank) | memorized wrong labels | [nlp.stanford.edu/sentiment](https://nlp.stanford.edu/sentiment/), loaded as [`nyu-mll/glue`](https://huggingface.co/datasets/nyu-mll/glue), config `sst2` |

The planted names are added by our own builders (`src/make_data.py` and the other `make_*.py` scripts); each built dataset
has a `manifest.json` with sha256 hashes so a rebuild can be verified.

## Where everything is

| Part of the study | Code | Report |
| --- | --- | --- |
| Data, training, checks, logit lens (GPT-2) | `make_data.py`, `train.py`, `checks.py`, `logit_lens.py`, `run_pilot.py` | `RESULTS.md` |
| Block swaps and single-block retraining | `swap.py`, `run_swap.py`, `retrain.py`, `run_night2.py`, `run_budget.py` | `RESULTS.md` |
| Repair without the names (control) | `retrain.py`, `report_nonames.py`, `colab/phase3c.ipynb` | `RESULTS_PHASE3C.md` |
| Activation patching; patching between repaired and unrepaired models | `patching.py`, `run_patching.py`, `report_restore.py`, `colab/phase_4b.ipynb` | `RESULTS_RESTORE.md` |
| LoRA (rank 4/64) and the weak fix (small adapter) | `train_lora.py`, `run_lora.py`, `weakfix.py`, `report_lora.py` | `RESULTS_LORA.md` |
| Repair-speed test on new seeds | `weakfix_newseeds.py`, `report_newseeds.py`, `colab/phase_l7.ipynb` | `RESULTS_L7.md` |
| Read-out from an intermediate layer, with controls | `reinject.py`, `run_reinject.py`, `run_reinject_controls.py` | `RESULTS_REINJECT.md`, `RESULTS_REINJECT_CONTROLS.md` |
| Pythia-160M | `make_pythia_data.py`, `train_pythia.py`, `logit_lens_pythia.py`, `swap_pythia.py`, `swap2_pythia.py`, `retrain_pythia.py` | `RESULTS_PYTHIA_PHASE2B.md`, `RESULTS_PYTHIA_PHASE3.md` |
| Word cue (film vs movie) | `make_wordcue.py`, `run_wordcue.py`, `run_wordcue_nc.py` | `RESULTS_WORDCUE.md`, `RESULTS_WORDCUE_NC.md` |
| Yelp | `make_yelp.py`, `run_yelp.py`, `report_yelp.py` | `RESULTS_YELP.md` |
| Memorized wrong labels (SST-2) | `make_wronglabel.py`, `train_wronglabel.py`, `run_w2.py`, `report_wronglabel.py` | `RESULTS_WRONGLABEL.md` |

All scripts are in `src/`; raw per-seed outputs are in `results/`; configs in `configs/`; tests in `tests/`.
[`SUMMARY.md`](SUMMARY.md) gives a generated overview of all results, and [`PROJECT_SPEC.md`](PROJECT_SPEC.md) holds every
pre-registered plan and decision rule.

## Reproducing the paper

The paper's numbers and figures are post-processing only: they read the committed `results/*.json` and run no model.

    python src/paper_numbers.py         # writes paper/numbers.tex (LaTeX macros) and PAPER_NUMBERS.md (value -> source file)
    python src/make_paper_figures.py    # writes the vector figures in figures/paper/
    python -m unittest tests/test_paper_numbers_tex.py

Then compile `paper/main.tex` with pdflatex + bibtex (IEEEtran class).

## Running the experiments

The experiments were run on Windows with Python 3.11 and an RTX 4050 laptop GPU (some phases on Colab, see `colab/`).
The commands below use Windows paths; on Linux or macOS use `python src/<script>.py`. Details of each phase, including
its decision rules, are in `PROJECT_SPEC.md`.

Setup (Windows, Python 3.11):

    py -3.11 -m venv .venv
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\python -m pip install transformers datasets numpy pandas matplotlib scikit-learn tqdm pyyaml

requirements.txt is the exact frozen set (torch is the cu128 build).

Reproduce the pilot (all steps run offline once gpt2 and imdb are in the HF cache):

    .venv\Scripts\python src\make_data.py        # rebuild; must match the sha256 in data/manifest.json (--verify = fast re-check)
    .venv\Scripts\python src\train.py --smoke    # 50-step smoke test
    .venv\Scripts\python src\run_pilot.py        # 6 runs + checks + logit lens + figures + RESULTS.md

Pieces can also be re-run alone: src/checks.py --run <run> | --base, src/logit_lens.py [--run <run>], src/report.py.
Log: logs/run_pilot.log.

Phase 2 (removal map, block swaps between the saved models):

    .venv\Scripts\python src\run_swap.py      # swap.py (resumable) + report.py; log: logs/run_swap.log

Night 2 (Phase 2b follow-up swaps + Phase 3 targeted retraining):

    .venv\Scripts\python src\run_night2.py     # swap.py --phase2b, retrain.py (gate first), report.py; log: logs/run_night2.log

Phase 5 (word-choice shortcut, film vs movie; PROJECT_SPEC.md "Phase 5"):

    .venv\Scripts\python src\run_wordcue.py   # make_wordcue.py (+ --verify), 6 trainings + checks (gate, 10% fallback),
                                              # lens, swaps, retrain (gate), patching, RESULTS_WORDCUE.md; log: logs/wordcue.log

Run it detached (PowerShell, from the repo folder; keeps running when the window is closed):

    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src\run_wordcue.py" -WorkingDirectory "$PWD" -WindowStyle Hidden -RedirectStandardOutput "logs\wordcue.stdout.txt" -RedirectStandardError "logs\wordcue.stderr.txt"
    Get-Content logs\wordcue.log -Wait -Tail 20             # follow it
    taskkill /PID (Get-Content logs\wordcue.pid) /T /F      # stop it (resumes when started again)

Phase 6 (mid-layer reinjection, inference only; PROJECT_SPEC.md "Phase 6"; after Phase 5):

    .venv\Scripts\python src\run_reinject.py  # reinject.py for full FT seeds 0-2, then wordcue seeds 0-2 if all 6 Phase 5
                                              # checks pass, then RESULTS_REINJECT.md; log: logs/reinject.log

Run it detached (PowerShell, from the repo folder):

    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src\run_reinject.py" -WorkingDirectory "$PWD" -WindowStyle Hidden -RedirectStandardOutput "logs\reinject.stdout.txt" -RedirectStandardError "logs\reinject.stderr.txt"
    Get-Content logs\reinject.log -Wait -Tail 20            # follow it
    taskkill /PID (Get-Content logs\reinject.pid) /T /F     # stop it (finished seeds are skipped on restart)

Offline test (no downloads): `python -m unittest tests/test_reinject.py`

Phase 6b (controls for Phase 6: equal dose, wider alpha grid, matched random control, cue-free alpha, sentiment part
of s; inference only; PROJECT_SPEC.md "Phase 6b"):

    .venv\Scripts\python src\run_reinject_controls.py  # reinject_controls.py for full FT seeds 0-2, then
                                                       # RESULTS_REINJECT_CONTROLS.md; log: logs/reinject_controls.log

Run it detached (PowerShell, from the repo folder):

    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src\run_reinject_controls.py" -WorkingDirectory "$PWD" -WindowStyle Hidden -RedirectStandardOutput "logs\reinject_controls.stdout.txt" -RedirectStandardError "logs\reinject_controls.stderr.txt"
    Get-Content logs\reinject_controls.log -Wait -Tail 20     # follow it
    taskkill /PID (Get-Content logs\reinject_controls.pid) /T /F   # stop it (finished seeds are skipped on restart)

Offline test (no downloads): `python -m unittest tests/test_reinject_controls.py`

Wrong-label memorization (W2):

- [Results and interpretation](WRONGLABEL_REPORT.md) and [per-seed intervention tables](RESULTS_WRONGLABEL.md).
- [Source verification and reproduction check](results/wronglabel/W2_TRACEABILITY.md), including the command for checking saved source hashes against the two recorded commits.
- [Wrong-label comparison figure](figures/paper/fig_f_wronglabel.pdf), generated from the committed swap and retraining results.

Paper figures and numbers audit (post-processing only: reads committed results and data manifests, runs no model):

    .venv\Scripts\python src\make_paper_figures.py   # vector PDFs in figures/paper/ (IEEE column, 3.45 in, serif 8 pt);
                                                     # a figure whose result files are missing is skipped and logged
                                                     # in figures/paper/make_paper_figures.log
    .venv\Scripts\python src\paper_numbers.py        # PAPER_NUMBERS.md: every per-seed number the paper uses, with
                                                     # its source file and JSON key, and paper/numbers.tex: the same
                                                     # values as LaTeX macros (\gapFullFTsZero{} = one seed,
                                                     # \gapFullFT{} = 0.646/0.626/0.541), checked equal to the table

Offline test (no downloads; the compile check is skipped without pdflatex): `python -m unittest tests/test_paper_numbers_tex.py`
