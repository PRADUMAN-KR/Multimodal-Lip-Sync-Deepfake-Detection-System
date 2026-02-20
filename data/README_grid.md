# GRID corpus (lip-reading / audiovisual)

## Download in one go

From the project root:

```bash
pip install zenodo_get   # one-time
python3 scripts/download_grid_corpus.py
```

Data is saved under `data/grid_corpus/` (~16 GB). Zips are extracted automatically.

To only print direct URLs (e.g. for a download manager or curl):

```bash
python3 scripts/download_grid_corpus.py --urls
```

## What you get (Zenodo version)

- **audio_25k.zip** – 25 kHz WAV per speaker (s1–s34), 1000 sentences each
- **alignments.zip** – word-level time alignments
- **s1.zip … s34.zip** (no s21) – video as sequences of .jpg frames per sentence
- **jasagrid.pdf** – corpus description

34 speakers, 1000 sentences each; sentences are like “put red at G9 now”. Good for lip-sync / lip-reading pipelines.

## Alternative: Sheffield (MPG video)

For original **.mpg** video (one file per sentence) instead of frame folders, use the official site and download per speaker:

- https://spandh.dcs.shef.ac.uk/gridcorpus/
- Video “normal”: ~480 MB per speaker (s1.mpg_vcd.zip, etc.)

The script above uses Zenodo for a single bulk download; Sheffield is manual per-speaker.

---

## Run lip-sync on GRID Sheffield (e.g. s1)

After you have extracted GRID Sheffield into `data/grid_sheffield/s1/` (or another folder of `.mpg` files):

```bash
# From project root; requires trained weights at weights/best_model.pth
python3 scripts/run_grid_eval.py

# Limit to 20 clips, or write results to CSV
python3 scripts/run_grid_eval.py --dir data/grid_sheffield/s1 -n 20 -o results/s1_results.csv
```

The pipeline reads each `.mpg` (video + audio extracted automatically) and runs lip-sync detection.
