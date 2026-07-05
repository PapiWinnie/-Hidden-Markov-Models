# Human Activity Recognition with Hidden Markov Models

Formative 2, Machine Learning Techniques II

A Hidden Markov Model, implemented from scratch in numpy, that classifies four activities (standing, walking, jumping, still) from smartphone accelerometer and gyroscope data. Built around the idea that a phone's built-in sensors, without any dedicated wearable, might be enough to tell these apart.

## Report

Full report (background, methodology, results, discussion): [Google Doc](https://docs.google.com/document/d/1W04N1O0vBtjQfgOXt7a2ZxSYkRBvQB0owyZ0fUxDmI0/edit?usp=sharing)

## Repository Contents

| File | Description |
|---|---|
| `hmm_activity_recognition.ipynb` | Full pipeline: data loading, feature extraction, HMM training, evaluation |
| `data/Formative_II_HMM/` | 51 raw Sensor Logger recordings (zip per recording) |
| `HMM_Activity_Recognition_Report.html` | Formatted report with embedded figures |
| `README.md` | This file |

## Data Collection

- App: Sensor Logger (Accelerometer + Gyroscope)
- Activities: standing, walking, jumping, still
- 51 recordings total: 13 standing, 13 walking, 12 jumping, 13 still
- Each recording 5 to roughly 14 seconds
- Sampling rate: configured for 50 Hz, but measured directly from each recording's `Metadata.csv` at 100 Hz (10 ms between samples). The actual measured rate is what the pipeline uses, not the configured one.

## Methodology

**Preprocessing**
- Accelerometer and gyroscope CSVs (delivered separately per recording) merged on nearest timestamp, since each sensor is polled independently
- Signal split into overlapping 2-second windows (200 samples, 50% overlap)
- 20 features per window: mean, standard deviation, and variance per accelerometer axis, signal magnitude area, accelerometer axis correlation (x-y, x-z), mean and standard deviation per gyroscope axis, and two FFT-derived features (dominant frequency, spectral energy) from the accelerometer x-axis
- Features z-score normalized, fit on training data only
- Train/test split done at the recording level (not window level), holding out the two most recent recordings per activity as a fully unseen test set

**Model**
- Gaussian-emission HMM with diagonal covariance, 4 hidden states (one per activity)
- Implemented from scratch: forward-backward, Viterbi decoding, Baum-Welch training
- Baum-Welch stops on a log-likelihood convergence check (change between iterations below tolerance), not a fixed iteration count
- State means initialized from labelled training data rather than randomly, for faster and more stable convergence

## Results

Training converged in 8 iterations. Overall accuracy on the 8 held-out test recordings: **89.3%**.

| Activity | Test Samples | Sensitivity | Specificity |
|---|---|---|---|
| Standing | 10 | 0.600 | 0.957 |
| Walking | 13 | 0.846 | 0.977 |
| Jumping | 16 | 1.000 | 1.000 |
| Still | 17 | 1.000 | 0.923 |

Jumping and still were classified perfectly, they sit at opposite ends of the motion-intensity spectrum. Standing was hardest to distinguish, mostly confused with still, likely because both are low-motion states that the current feature set doesn't fully separate on subtle postural sway alone.

A separate test, concatenating four unseen recordings (one per activity) into a continuous sequence and decoding across all of it, reached 93.3% accuracy and correctly caught each activity transition with only a short lag.

## Running the Notebook

1. Place your folder of recording zips (or a zip of that folder) at `data/Formative_II_HMM` relative to the notebook, or update `DATA_PATH` at the top of the data loading cell
2. Run all cells top to bottom
3. If `DATA_PATH` isn't found, the notebook falls back to synthetic data so it still runs standalone

## Possible Improvements

- A feature more sensitive to subtle postural sway (e.g. gyroscope variance over a longer window), to better separate standing from still
- More recordings across more people, phones, and environments, everything here came from one person and one device
- Full covariance matrices instead of diagonal, to capture correlations between accelerometer and gyroscope features
