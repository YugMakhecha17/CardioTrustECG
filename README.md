# CardioTrust — A Trustworthy Deep Learning for 12-Lead ECG Diagnosis

**a simple one line pitch on why i made this:** an ECG classifier that doesn't just predict — it *knows when it is unsure*, *shows where it is looking*, and *has been audited for robustness and fairness*.

| Stage | What it demonstrates |
|---|---|
| 1. Baseline | 1D-ResNet on the official PTB-XL patient-level folds (no leakage) |
| 2. Main model | CNN-Transformer hybrid + **self-supervised contrastive pre-training** |
| 3. Label efficiency | SSL vs. from-scratch at 5% / 10% / 25% / 100% of labels |
| 4. Calibration | Temperature scaling, ECE, reliability diagrams |
| 5. Uncertainty | **Conformal prediction** (class-conditional coverage guarantee) + clinician-deferral triage |
| 6. Explainability | 1D Grad-CAM + a quantitative **deletion test** (not just pretty pictures) |
| 7. Audit | Subgroup (sex / age), noise and lead-dropout robustness |
| 8. Export | Weights + calibration + thresholds, ready for a FastAPI service |

**Dataset:** [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet, open access, ~21.8k 10-second 12-lead ECGs, 18.9k patients). Task: multi-label classification into 5 diagnostic superclasses — `NORM, MI, STTC, CD, HYP`.

**How to run:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. Set `QUICK = True` in the config cell first for a fast end-to-end smoke test (very low accuracy, just checks that everything runs).

> ⚠️ **Research prototype only.** Not a medical device, not clinically validated, not for diagnostic use.

## · Data
i used the **100 Hz** version (1000 samples × 12 leads per record) and the dataset authors' recommended split, which is stratified **by patient** so no patient appears in two folds:
`folds 1–8 → train`, `fold 9 → validation (model selection + temperature scaling)`, `fold 10 → test`.

## · Models & training utilities
* **`ResNet1D`** – strong, standard baseline.
* **`CNNTransformer`** – a 3-layer strided conv stem tokenises the ECG (1000 → 125 tokens), followed by a pre-norm Transformer encoder with a learned `[CLS]` token and positional embeddings.
* **Augmentations** are physiologically motivated: amplitude scaling, Gaussian noise, baseline wander (0.1–0.5 Hz), small time shifts, lead dropout and time masking.

## · Self-supervised pre-training (SimCLR-style contrastive learning)
The encoder learns from **unlabelled** training ECGs only: two augmented views of the same ECG are pulled together in embedding space, all other ECGs in the batch are pushed away (NT-Xent loss). Validation and test folds are never touched.

## · Label-efficiency experiment — does pre-training help?
Same architecture, same recipe; the only difference is the initialisation (random vs. SSL-pretrained). Small label fractions get proportionally more epochs so both arms see a comparable number of gradient steps.

## · Calibration
A model's probabilities should mean what they say: of all predictions at "80%", about 80% should be positive. We fit one temperature `T` on the validation fold (`p = σ(logit / T)`) and report **Expected Calibration Error** before and after.

## · Conformal prediction & clinician deferral
For each diagnosis `c` we use **split conformal prediction** with score `1 − p_c` on calibration ECGs that truly have `c`. The resulting per-class threshold guarantees, under exchangeability,

> `P( c ∈ prediction set | patient truly has c ) ≥ 1 − α`

i.e. a **guaranteed minimum recall per diagnosis** (class-conditional coverage). Calibration data must be independent of the evaluation data, so we split **test fold 10 by patient** into a calibration half and an evaluation half.

Coverage is guaranteed *on average over calibration draws*; a single split can land a few points off target (more so for rare classes), so we also repeat the split 30 times.

**Triage rule:** a case is *confident* if the prediction set is non-empty and not self-contradictory (`NORM` together with a pathology). Everything else is **deferred to a clinician**.

## · The CRUX: Explainability — 1D Grad-CAM
Gradients of a diagnosis logit are taken w.r.t. the conv-stem feature map (the "tokens" fed to the Transformer), turned into a relevance profile over time and up-sampled to the 10-second signal. We then **test** the explanations quantitatively instead of just showing them.

## · Robustness & fairness audit
Headline AUROC hides failure modes. We break performance down by **sex** and **age band**, then stress-test against **sensor noise** and **missing leads** (a realistic failure with wearables / loose electrodes), comparing all three models.

## Limitations & honest caveats
* **Single-source data:** PTB-XL comes from one German device/centre network. External validation (e.g. on Chapman-Shaoxing, CPSC 2018, or PhysioNet/CinC 2021 data) is the most valuable next step and is not done here.
* **Superclass labels only** (5 classes); fine-grained rhythm/morphology statements are out of scope.
* **Labels are noisy** (automatic + cardiologist annotations with likelihoods); AUROC ceilings are partly label-noise ceilings.
* **Conformal guarantees are marginal** per class under exchangeability with the calibration data — they do not hold under distribution shift, which is exactly why the noise/lead audits matter.
* **Grad-CAM is a heuristic.** The deletion test shows the highlighted regions matter to the model; it does not prove they match clinical reasoning.
* **Not a medical device.** Research and education only.

## Next steps
1. Serve the exported artifacts behind a **FastAPI** endpoint (predict → calibrated probabilities → conformal set → saliency image → defer/auto flag).
2. External validation on a second ECG dataset + subgroup re-audit.
3. Add an LLM-generated plain-language report strictly grounded in the model's structured outputs.
