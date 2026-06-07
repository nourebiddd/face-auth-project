# Face Authentication: Real vs. Synthetic Enrollment Data

> A pilot study comparing face authentication systems enrolled on real photographs versus AI-generated synthetic images, evaluated under NIST FRTE-aligned metrics.

**Author:** Noureldin Ebid  
**Course:** Seminar zu aktuellen Entwicklungen (M14) · M.Sc. Angewandte Informatik · HTW Berlin · SoSe 2026  
**Supervisor:** Prof. Dr. Andrea Knaut

---

## Overview

Face authentication systems traditionally store real facial photographs of every enrolled user — data classified as sensitive personal data under **GDPR Article 9**, exposing organisations to breach liability and compliance obligations.

This project investigates a privacy-preserving alternative: enrolling users using **category-(b) synthetic images** — algorithmically derived from real photographs through a controlled lossy pipeline, such that no original biometric data is retained in the authentication database.

Two experiments are run under identical, balanced evaluation conditions:

| | Experiment 1 | Experiment 2 |
|---|---|---|
| **Enrollment data** | Real photographs | Category-(b) synthetic images |
| **Model** | FaceNet (DeepFace, pre-trained) | FaceNet (DeepFace, pre-trained) |
| **Threshold θ** | 10.0 | 10.0 |
| **Test set** | Shared (L₁ = L₀ = 20) | Shared (L₁ = L₀ = 20) |

> ⚠️ **Pilot scale only.** With L₁ = L₀ = 20 queries, each single decision = 5 pp swing. Results are exploratory observations, not statistically reliable findings.

---

## What Is "Synthetic" Here?

The term is used inconsistently in the literature. This study adopts a precise operational definition:

| Category | Description | Used as |
|---|---|---|
| **(a) Fully AI-generated** | No correspondence to any real person (e.g. StyleGAN2, diffusion models) | Impostor queries |
| **(b) Controlled pipeline derivatives** | Derived from real photos via a lossy multi-stage pipeline — original pixel-level biometric data not recoverable | Enrollment images |

The GDPR relevance: the critical question is not whether the image *looks* different, but whether the original biometric data is *reconstructible*. Category (b) eliminates recoverable biometrics from the stored database.

---

## Results (Pilot Scale)

| Metric | Exp 1 (Real) | Exp 2 (Synthetic) | Δ |
|---|---|---|---|
| **FMR** | 15% (3/20) | 5% (1/20) | −10 pp |
| **FNMR** | 10% (2/20) | 15% (3/20) | +5 pp |
| **Accuracy** | 87.5% (35/40) | 90.0% (36/40) | +2.5 pp |
| **d′** | 2.41 | 2.18 | −0.23 |

### Score Distributions

| | Genuine μ | Genuine σ | Impostor μ | Impostor σ | d′ |
|---|---|---|---|---|---|
| Exp 1 (Real) | 7.21 | 2.53 | 11.84 | 1.18 | 2.41 |
| Exp 2 (Synthetic) | 8.43 | 2.01 | 12.61 | 0.74 | 2.18 |

### Why the Trade-off Happens

The 5-stage pipeline controls pose, lighting, and colour → embeddings cluster more tightly around the identity centroid.

- **FMR decreases ✓** — tighter centroid is more precisely separated from impostor embeddings; impostors are less likely to fall inside the acceptance radius
- **FNMR increases ✗** — tighter cluster may not cover the full natural appearance variation of the user; genuine test photos in different lighting/expression can land just outside the boundary

This is the classical FMR/FNMR trade-off described in Jain et al. (2025), Section 1.4.

---

## System Architecture

### Registration

```
real photo → FaceNet → 128-dim embedding
                              ↓
              mean of 10 enrollment embeddings
                              ↓
                     reference vector ē_u
                              ↓
                       JSON database
```

### Authentication

```
query photo → FaceNet → embedding e_q
                              ↓
              L2 distance to nearest ē_u
                              ↓
          d < θ → VERIFIED ✓
          d ≥ θ → REJECTED ✗
```

---

## The 5-Stage Synthetic Pipeline (Experiment 2)

Each real enrollment photo is transformed into synthetic outputs through 5 controlled stages. The originals are then discarded — only FaceNet embeddings of the synthetic outputs are stored.

```
Real Photo
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Alignment                                            │
│  InsightFace detects 5 keypoints → canonical 256×256 frame     │
│  Eliminates pose variation between images                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Lighting                                             │
│  Gamma correction γ ~ U(0.55, 1.60) via lookup table           │
│  γ < 1 → brighter (overcast)  ·  γ > 1 → darker (artificial)  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Colour Temperature                                   │
│  Warm / cool / neutral mode, equal probability                  │
│  RGB scale factors from [1.05, 1.20] and [0.80, 0.95]          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — Skin Tone Normalisation                              │
│  CLAHE on L channel of CIE LAB space, clip limit ~ U(1.5, 3.5) │
│  Normalises luminance without hue shift                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — Camera Quality & Pose                                │
│  Gaussian blur k ∈ {3,5} (50% prob) · noise σ ~ U(2,8)        │
│  In-plane rotation ~ U(−12°, 12°)                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Synthetic Image → FaceNet → embedding stored
Original Photo  → discarded
```

### Rejected Prior Approaches

Two face-swap approaches were implemented and rejected before the pipeline above:

**Approach A — Blind face swap:** Poisson-cloned the aligned face onto random StyleGAN2 base images. ~50% of the base pool depicted female subjects → gender mismatch caused FNMR to reach 75%.

**Approach B — Gender-filtered face swap:** Restricted base pool to male-presenting subjects (InsightFace confidence > 0.6). Poisson blending boundary artefacts created a persistent 1–2 unit upward bias in genuine L2 scores.

**Lesson:** Any uncontrolled variable in the synthesis process propagates into the embedding space and corrupts the results.

---

## Dataset

| Split | Contents | Size |
|---|---|---|
| Enrollment | First 10 real photos per user (sorted lexicographically) | 40 images |
| Genuine test (L₁) | Last 5 real photos per user — never seen during enrollment | 20 queries |
| Impostor test (L₀) | StyleGAN2 AI-generated strangers (thispersondoesnotexist.com) | 20 queries |

- **Enrolled users:** amr, adel, otto, mazen (4 male adults)
- **Capture conditions:** Natural, uncontrolled — smartphone camera, varying lighting, background, pose
- **Test set is identical for both experiments** — any metric difference is attributable solely to enrollment data type

---

## Tech Stack

| Component | Tool |
|---|---|
| Face embedding | [FaceNet](https://arxiv.org/abs/1503.03832) via [DeepFace](https://github.com/serengil/deepface) |
| Face alignment | [InsightFace](https://github.com/deepinsight/insightface) (`buffalo_l`) |
| Skin tone normalisation | CLAHE ([Zuiderveld, 1994](https://dl.acm.org/doi/10.5555/180895.180940)) |
| Impostor generation | [StyleGAN2](https://arxiv.org/abs/1912.04958) via thispersondoesnotexist.com |
| Evaluation framework | [NIST FRTE](https://www.nist.gov/programs-projects/face-recognition-technology-evaluation-frte) · [Jain et al. (2025)](https://link.springer.com/book/9783031624100) |

---

## Limitations

- **Sample size:** 20 genuine + 20 impostor queries. Each decision = 5 pp swing. No meaningful confidence intervals possible.
- **Single operating point:** θ = 10.0 selected empirically — not from a DET curve at the EER. No full DET curve produced.
- **Demographics:** All 4 users are male adults. Results do not generalise to female subjects, older adults, or cross-ethnic settings.
- **Synthetic impostors only:** All impostor queries are StyleGAN2 faces — no real human impostors tested.
- **Still requires real photos:** The pipeline transforms real photos; it cannot generate a user from scratch. A real photo is still needed at enrollment time.
- **No statistical testing:** Formal hypothesis testing is not meaningful at this scale. Results are observations to motivate future work.

---

## Future Work

- [ ] Full DET curve evaluation — sweep θ from 1 to 20, compute EER and TAR@FMR
- [ ] Scale to 20+ users, 50+ genuine test images per user, 500+ impostor attempts (NIST FRTE minimum)
- [ ] Fully generative per-identity synthesis (e.g. IP-Adapter / diffusion) — zero real photos at enrollment
- [ ] Demographic diversity — female subjects, older adults, cross-ethnic settings
- [ ] Real human impostor evaluation
- [ ] Multi-model comparison — ArcFace, VGGFace2

---

## Ethical Considerations

- All 4 enrolled users provided informed consent for use of their images in this academic research project
- All photographs stored locally only — no biometric data shared with any third-party service
- Processing falls under the GDPR Article 89 research exemption
- All 20 impostor images are entirely AI-generated — no real third-party photographs used without consent

---

## References

1. Jain, A. K., Ross, A. A., Nandakumar, K., & Swearingen, T. (2025). *Introduction to Biometrics* (2nd ed.). Springer.
2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *Proc. IEEE CVPR*, 815–823.
3. NIST. (2024). Face Recognition Technology Evaluation (FRTE). https://www.nist.gov/programs-projects/face-recognition-technology-evaluation-frte
4. Serengil, S. I., & Ozpinar, A. (2020). LightFace: A hybrid deep face recognition framework. *Proc. IEEE ASYU*.
5. Wood, E., et al. (2021). Fake it till you make it: Face analysis in the wild using synthetic data alone. *Proc. IEEE ICCV*.
6. Bae, G., et al. (2023). DigiFace-1M: 1 million digital face images for face recognition. *Proc. IEEE WACV*.
7. Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. *Graphics Gems IV*, Academic Press, 474–485.
8. European Parliament and Council. (2016). Regulation (EU) 2016/679 — GDPR. *Official Journal of the EU*, L 119, 1–88.
9. Goodfellow, I., et al. (2014). Generative adversarial nets. *NeurIPS*, vol. 27.
10. Karras, T., et al. (2020). Analyzing and improving the image quality of StyleGAN. *Proc. IEEE CVPR*, 8107–8116.
11. Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *Proc. IEEE CVPR*, 10684–10695.

---

*HTW Berlin · Fachbereich 4 · M.Sc. Angewandte Informatik · SoSe 2026*
