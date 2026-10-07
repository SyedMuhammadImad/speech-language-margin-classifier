# Speech-language margin classification

Explicit binary hinge-loss optimization extended to multiclass one-versus-rest classification, compared with library linear and RBF support-vector classifiers on locally retained WAV recordings. See `ATTRIBUTION.md` for shared source/data provenance and assisted changes. The implementation is finished as an experimental comparison; its measured transfer to unseen speakers is weak.

## Run

Use Python 3.12, install `requirements.txt`, then run:

```powershell
python -m pytest -q -p no:cacheprovider
python experiment.py train --data-dir path/to/recordings --cache-dir local/cache --output-dir local/results
python experiment.py predict --checkpoint local/results/fold1-manual.npz --wav local/clip.wav
```

Folders can be named `speaker-a english`, `speaker-a urdu`, and `speaker-a mixed`; other layouts use `--manifest manifest.csv` with relative `path,label,speaker` columns. At least three speakers are required, each containing every class. Inputs must be distinct finite non-silent mono/stereo WAV recordings, 0.15–60 seconds and under 32 MB. Conflicting duplicates and traversal fail explicitly. Only the first 30 seconds are resampled to mono 16 kHz. Thirteen mean MFCCs use FFT 512 and hop 256. Raw audio, feature caches, fitted models, plots and other media remain local.

## Optimization and evaluation

For a binary classifier the objective is `lambda * ||w||² / 2 + mean(max(0, 1 - y*(Xw+b)))`. The implementation retains explicit shuffled per-example SGD updates. A zero binary margin maps to the first sorted class. Three one-versus-rest models choose the largest margin; margins are uncalibrated scores, not probabilities. Arbitrary integer/string target labels are decoded by the classifier.

Each of three folds uses one speaker for training, another for validation and the third for testing. Scaling fits the training speaker only. Validation macro F1 selects regularization from 0.001/0.01/0.1 for the manual model and C from 0.1/1/10 for each library kernel. Test-speaker data does not select settings. The 268-recording dataset contains English, Urdu and mixed speech. Results pooled over the disjoint test speakers:

- manual_linear: 35.82% accuracy, 0.358 macro F1
- sklearn_linear: 37.69% accuracy, 0.345 macro F1
- sklearn_rbf: 34.33% accuracy, 0.324 macro F1

This is close to chance performance and does not support a reliable language-detection product. Three voices and possible shared phrases/channel conditions limit interpretation. Mixed speech is a clip label, not word-level code-switch detection. These results do not reproduce the earlier larger four-language dataset.

## Saved inference and verification

Each manual checkpoint retains weights, biases, class IDs and the training scaler. Inference applies that scaler without refitting. The CLI returns an encoded class and uncalibrated margins; `metrics.json` supplies the encoded label order. The experiment verifies every held-out prediction/margin after reload and runs one actual WAV through every reloaded checkpoint. No model binary is included here; reproduce training locally before invoking inference.

Six independent checks cover finite-difference and autograd agreement for the hinge gradient, negative/string labels, zero-margin ties, reproducibility, invalid inputs, scaler/checkpoint parity, speaker partitions and malformed-file/CLI failure behavior. The notebook follows setup → implementation → results and has been executed with outputs stripped. Numeric metrics are retained. Results and source attribution remain explicit; established algorithms are not original research.
