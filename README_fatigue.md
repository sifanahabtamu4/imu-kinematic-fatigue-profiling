# imu-kinematic-fatigue-profiling

A dual-output CNN–LSTM that simultaneously classifies **which** upper-limb rehabilitation exercise is being performed and assesses **how well** it is performed, from a single wrist-worn IMU — evaluated under strict leave-one-subject-out cross-validation.

Implementation accompanying:

> S. T. Araya, S. H. Eshetu, A. M. Endale, A. A. Damtie, A. B. Balcha, Y. Benachour and A. Natsheh, "Characterizing Proximal Kinematic Fatigue Profiles Using a Standalone IoMT Wrist Sensor," 8th HCT International Multi-Conference on Advances in Science and Engineering Technology (ASET), 2026.

---

## Results

Every figure below is measured on participants held entirely out of training, across **15 subjects** and **3,001 repetitions** of a **12-exercise** protocol.

| Metric | Value |
|---|---|
| Exercise classification accuracy | **88.73% ± 6.00%** |
| Exercise classification macro-F1 | 82.87% ± 13.78% |
| Execution quality accuracy | **88.99% ± 4.66%** |
| Execution quality ROC-AUC | **0.9060 ± 0.0553** |

### Ablation — read this before citing the headline

| Architecture | Taxonomy acc. | Taxonomy F1 | Quality ROC-AUC |
|---|---|---|---|
| DTW 1-nearest-neighbour | 88.56% | **0.8344** | — |
| LSTM-only | 60.06% | 0.5418 | 0.7175 |
| CNN-only | **88.96%** | 0.8445 | 0.8639 |
| Proposed CNN–LSTM | 88.73% | 0.8287 | **0.9060** |

**On exercise taxonomy, the hybrid architecture is not the best model here.** A CNN-only variant reaches 88.96% and a non-parametric DTW 1-nearest-neighbour classifier reaches 88.56%, both with higher macro-F1 than the full model's 88.73% / 0.8287. Spatial feature extraction alone is sufficient to decide which exercise is being performed, and the LSTM contributes nothing measurable to that task.

**The hybrid earns its place on execution quality**, where the LSTM lifts ROC-AUC from 0.8639 to 0.9060 — roughly 0.042. Detecting *how well* a movement is performed requires modelling how it evolves over time in a way that identifying *which* movement it is does not. The LSTM-only collapse to 60.06% shows the reverse: a recurrent layer without convolutional filtering cannot separate the classes at all.

So the defensible claim is narrow: **the architecture is justified by the quality task, not the taxonomy task.** DTW 1-NN is also competitive on taxonomy while offering no quality assessment at all and requiring a comparison against the entire training set at inference — which is the wrong shape for a wrist-worn device.

---

## What makes the evaluation strict

1. **Leave-one-subject-out.** No participant appears in both training and test. These are subject-independent numbers, not held-out sessions from known users.
2. **Per-fold ground truth.** The DTW barycentre ("kinematic centroid") and the `StandardScaler` are fitted inside each training fold only, then applied unchanged to the held-out subject.
3. **Augmentation after the split.** The training subset is isolated first, then augmented 3×, so no augmented clone of a repetition can land on both sides of the train/validation boundary. The execution order is `split → augment → derive features → scale`.
4. **Segmentation respects boundaries.** Repetition segmentation runs inside `groupby(['Subject_ID', 'Exercise'])`, so no window straddles two participants or two exercises.

## How the quality labels are made — and what that means

There are no therapist-annotated quality labels in this study. For each exercise class, a multivariate DTW barycentre is computed over the training repetitions; each repetition's DTW distance to that centroid is measured; repetitions beyond **mean + 1.5σ** are labelled *degraded*.

Two consequences, both acknowledged in the paper:

- These are **pseudo-labels** — an unsupervised proxy for kinematic deviation, not clinical ground truth. The network is trained to reproduce a DTW threshold rule and cannot outperform the rule that generated its targets. Validation against EMG or therapist-annotated video is named as future work.
- The centroid for a held-out participant comes from *other people*. A participant with idiosyncratic but entirely consistent form will be flagged against a population they do not resemble.

**The number that matters most for deployment is not in the headline table:** precision on the *degraded* class is **0.542**. Roughly one in two degradation alerts would be a false positive. Any deployed system needs cost-sensitive classification or a tuned probability threshold before it starts sending correction prompts to patients.

---

## Known issues in this implementation

Disclosed rather than silently patched, because fixing them would change the reported numbers:

1. **No random seeds are set.** Neither TensorFlow weight initialisation nor the NumPy augmentation RNG is seeded; only `train_test_split` carries a `random_state`. A re-run will not reproduce 88.73% exactly. The ± values reported are across folds, not across seeds.
2. **Rotation augmentation is physically inconsistent.** The random rotation (≤ 8°) is applied to the accelerometer and gyroscope triads but not to Roll/Pitch/Yaw, which are orientation channels in the same frame. Augmented samples therefore carry orientation columns inconsistent with their rotated inertial columns — a mild implausible perturbation rather than a correctness bug, but a stricter implementation would rotate all three triads together.
3. **Resampling discards duration.** Every repetition is interpolated to 150 time steps, so absolute repetition duration is not available as a feature. This is deliberate — duration is partly a property of the person — but it does remove a signal that may carry fatigue information.
4. **Segmentation is tuned, not learned.** The 0.8 Hz low-pass cutoff, the 0.20 × σ dynamic prominence, the 0.8 s valley separation and the 0.5 s minimum duration were selected empirically on this cohort. They are unlikely to transfer unchanged to participants with tremor or severely restricted range of motion.

---

## Data

**The data is not in this repository and cannot be redistributed.** The recordings come from human participants under informed consent obtained in accordance with the Declaration of Helsinki and institutional ethics approval; that consent does not extend to public release.

To run the notebook, place the CSV at:

```
data/Rehab_Data_Full_Rebuilt.csv
```

Expected schema:

| Column | Notes |
|---|---|
| `Subject_ID` | Participant identifier, e.g. `Subject_01` … `Subject_15`. Used as the LOSO grouping key. |
| `Exercise` | One of the 12 exercise labels below. |
| `Roll`, `Pitch`, `Yaw` | Device orientation, degrees. |
| `RotationX`, `RotationY`, `RotationZ` | Gyroscope, rad/s. |
| `AccelerationX`, `AccelerationY`, `AccelerationZ` | Accelerometer, g. |

Sampling frequency is assumed to be **100 Hz** (`fs=100`, hard-coded in the segmentation and jerk calculations — change it there if your data differs).

Derived channels (acceleration magnitude, gyroscope magnitude, jerk) are computed by the notebook and should not be present in the input. Rows missing `Subject_ID` or `Exercise` are dropped.

The 12 exercises: Banded Overhead Reach, Bicep Curl, External Rotation, Forearm Pronation–Supination, Front Delt Raise, Lateral Delt Raise, One-Arm Tricep Extension, Scaption, Scapular, Ulnar Nerve Glide, Wand Extension, Wand Flexion.

---

## Running it

```bash
git clone https://github.com/<your-username>/imu-kinematic-fatigue-profiling.git
cd imu-kinematic-fatigue-profiling
pip install -r requirements.txt
jupyter notebook imu_kinematic_fatigue_profiling.ipynb
```

Run the cells in order. The LOSO loop trains 15 models, and the ablation cells train 30 more, so a full pass is the slow part — budget accordingly. The DTW barycentre computation (`dtw_barycenter_averaging`, 5 iterations per class per fold) is the other significant cost.

Figures are written to `figures/` at 300 dpi.

## Repository contents

```
imu_kinematic_fatigue_profiling.ipynb   Full pipeline: segmentation → normalisation →
                                        augmentation → dual-head training → LOSO →
                                        confusion matrices → neural and DTW ablations
requirements.txt                        Pinned dependencies
README.md                               This file
data/                                   Not included — see Data above
figures/                                Generated on run
```

---

## Method summary

**Segmentation.** 4th-order Butterworth low-pass at 0.8 Hz on the acceleration magnitude, then peak detection with dynamic prominence `max(σ × 0.20, 0.05)` and minimum peak separation of 1 s. Repetition boundaries are the surrounding valleys, separated by at least 0.8 s, with candidates shorter than 0.5 s discarded.

**Normalisation and features.** Linear interpolation to 150 time steps. Nine raw channels plus three derived (acceleration magnitude, gyroscope magnitude, jerk) give a 150 × 12 input tensor. The magnitude channels are orientation-invariant, which is the point — a wrist-worn device is re-donned at a different rotation every session.

**Augmentation**, training fold only, 2 synthetic copies per repetition: random rotation ≤ 8° on the accelerometer and gyroscope triads, Gaussian jitter σ = 0.03, cubic-spline magnitude warping σ = 0.15 with 4 knots, cubic-spline time warping σ = 0.15.

**Architecture.**

```
Input (150 × 12)
  └─ Conv1D(32, k=3, relu) → MaxPool(2) → Dropout(0.2)
       └─ LSTM(64) → Dropout(0.2)
            ├─ Dense(32, relu) → Dense(n_classes, softmax)   # taxonomy
            └─ Dense(32, relu) → Dense(1, sigmoid)           # quality
```

Adam; sparse categorical cross-entropy on the taxonomy head and binary cross-entropy on the quality head; batch size 32; up to 30 epochs. Balanced class weights computed separately per head and passed as sample weights. Early stopping on validation loss (patience 5, best weights restored) and `ReduceLROnPlateau` (factor 0.5, patience 3, floor 1e-5).

## Limitations

- **Healthy cohort only.** No participant had a pathological mobility restriction. Whether a single statically-thresholded distal sensor can segment and classify movement under adhesive capsulitis or severely restricted range of motion is an open question.
- **Quality labels are an unsupervised proxy**, not clinical ground truth. See above.
- **Degraded-class precision is 0.542** — a high false-positive rate that would need mitigating before patient-facing deployment.
- **Single distal sensor.** The premise of the study is that a wrist IMU can infer proximal (shoulder, elbow) compensation. That inference is indirect by construction.
- **15 participants.** Enough for leave-one-subject-out to be meaningful, not enough to characterise performance across age, BMI or sex.

## Future work

- Validate the pseudo-labels against EMG or therapist-annotated video.
- Extend to cohorts with pathological mobility restriction.
- Adaptive rather than static pre-processing thresholds, to accommodate tremor and micro-movements.
- Cost-sensitive classification or threshold tuning to reduce degraded-class false positives.

## Citation

```bibtex
@inproceedings{araya2026fatigue,
  title     = {Characterizing Proximal Kinematic Fatigue Profiles Using a
               Standalone {IoMT} Wrist Sensor},
  author    = {Araya, Saron Tesfaye and Eshetu, Sifana Habtamu and
               Endale, Amanuel Mulugeta and Damtie, Amanuel Adissu and
               Balcha, Abel Bekele and Benachour, Yassine and Natsheh, Ammar},
  booktitle = {8th HCT International Multi-Conference on Advances in Science
               and Engineering Technology (ASET)},
  year      = {2026}
}
```

## Contact

Sifana Habtamu Eshetu — [ORCID 0009-0008-7537-3959](https://orcid.org/0009-0008-7537-3959) · [Google Scholar](https://scholar.google.com/citations?user=MTokPWMAAAAJ) · [GitHub](https://github.com/sifanatiya4)

Division of Engineering Technology & Science, Higher Colleges of Technology, Dubai.

Related: [cnn-lstm-imu-motion-recognition](https://github.com/sifanatiya4/cnn-lstm-imu-motion-recognition) — upper-limb movement recognition from wrist-worn IMU data.

## License

Code released under the MIT License. The dataset is not released.
