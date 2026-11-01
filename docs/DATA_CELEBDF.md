# Celeb-DF v2: getting the data and building the manifest

## 1. Get the data

Request access on the official page, [Celeb-DF (v2) on GitHub](https://github.com/yuezunli/celeb-deepfakeforensics).
It says to fill in the Google or Tencent form there, and that a download link is sent once the request
is accepted. The data is released under the "Terms to Use Celeb-DF". **Read those terms yourself**; they
are not reproduced on the page and this repository cannot tell you what they allow. In particular,
check them before uploading the videos to any cloud service, and never commit the videos to git
(`data/` is git-ignored).

Per the official page the dataset has three folders: `Celeb-real` (590 videos), `YouTube-real`
(300 videos) and `Celeb-synthesis` (5,639 videos).

## 2. Expected layout and file names

```
Celeb-DF-v2/
  Celeb-real/        id0_0000.mp4 ...        id<N>_<NNNN>.mp4
  Celeb-synthesis/   id0_id1_0000.mp4 ...    id<N>_id<M>_<NNNN>.mp4
  YouTube-real/      00000.mp4 ...           <NNNNN>.mp4
```

The folder names and counts are from the official page. **The file naming is not stated there.** It is
taken from a third-party project's description of the dataset and matches the original scripts'
assumptions, but nobody has checked it against your copy. The builder only accepts these patterns
and stops with an error if a folder has no matching file, rather than guessing which person is in a video.

## 3. Build the manifest

```powershell
python -m src.make_manifest celebdf --root D:\data\Celeb-DF-v2 --out manifest.csv
```

Options: `--seed 42` (reproducible), `--ratios 0.6,0.2,0.2` (train, val, test share of identities).

**Identity-disjoint split.** The same people appear in many Celeb-DF videos, and each fake involves two
identities. The builder therefore divides the *identities* into train, val and test sets, and keeps a
video only if every identity in it belongs to the same set. Videos that mix identities from different
sets are dropped and counted. A test result then reflects people the model never saw, which is the
honest way to measure it. `YouTube-real` videos have no identity in their names, so they are divided by
video; the same person could in principle appear in two splits there.

Expect to lose a large share of the fakes. If fake pairs were spread evenly, about 44% would stay with
the default ratios and the rest would be dropped (an estimate, not a measurement). The command prints
the exact counts per split, and `manifest.csv.summary.json` records the seed, which identities went
where, and what was dropped. `fit` needs at least 20 real and 20 fake videos in `val`; the command warns
if that is not met.

The official `List_of_testing_videos.txt` is **not** used. Its format and identity separation could not
be verified, so numbers from this split are not comparable with published Celeb-DF results.

## 4. Check, train, evaluate

```powershell
python -m src.evaluate validate manifest.csv
python -m src.retrain prepare manifest.csv --cache cache_faces
python -m src.retrain train --cache cache_faces --out output_v2\deepfake_cnn_lstm_v2.pth --manifest manifest.csv
python -m src.evaluate score manifest.csv --out results\scores_clean.csv --checkpoint output_v2\deepfake_cnn_lstm_v2.pth
python -m src.evaluate fit results\scores_clean.csv --out models\decision_policy.json --calibrate
python -m src.evaluate report results\scores_clean.csv --policy models\decision_policy.json
```

Training needs a GPU in practice (free Colab or Kaggle sessions work). The `prepare` step reads only the
`train` and `val` splits; the `test` split is first read by `score` and is used only for the report.

## 5. What this will and will not show

- It measures one dataset on unseen identities. It does not show how the model handles other
  manipulation methods, other datasets, or videos re-encoded by social platforms.
- For a cross-dataset check, get a second dataset (for example FaceForensics++, which has its own access
  form), add its rows to the manifest by hand with a different `dataset` name and `split` set to `test`
  only, and the report will show it separately. There is no builder for it yet.
- Run `score` again with `--condition jpeg:30`, `scale:0.5` or `blur:2` to see how results degrade.
