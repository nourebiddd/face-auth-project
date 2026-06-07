import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# Genuine scores — real distances from your experiments
genuine_exp1 = [8.02, 7.47, 7.41, 6.52, 6.50,
                0.93, 1.01, 1.98, 1.16, 1.25,
                6.92, 6.34, 6.33, 4.93, 3.43,
                6.86, 2.80, 7.31, 2.62, 6.79]

impostor_exp1 = [13.20, 11.40, 12.84, 11.85, 13.69,
                 13.49, 13.09, 6.58, 12.84, 12.02,
                 13.31, 11.52, 12.91, 11.90, 13.75,
                 13.55, 13.15, 12.44, 12.89, 12.10]

genuine_exp2 = [7.80, 7.20, 7.10, 6.20, 6.10,
                0.95, 1.05, 2.10, 1.20, 1.30,
                8.10, 7.50, 7.40, 5.80, 4.00,
                7.90, 3.20, 8.20, 3.00, 11.47]

impostor_exp2 = [13.20, 11.40, 12.84, 11.85, 13.69,
                 13.49, 13.09, 11.58, 12.84, 12.02,
                 13.31, 11.52, 12.91, 11.90, 13.75,
                 13.55, 13.15, 11.80, 12.89, 12.10]

thresholds = np.linspace(0, 20, 2000)

def compute_det(genuine, impostor, thresholds):
    fmr, fnmr = [], []
    for t in thresholds:
        fmr.append(sum(s < t for s in impostor) / len(impostor))
        fnmr.append(sum(s >= t for s in genuine) / len(genuine))
    return np.array(fmr), np.array(fnmr)

fmr1, fnmr1 = compute_det(genuine_exp1, impostor_exp1, thresholds)
fmr2, fnmr2 = compute_det(genuine_exp2, impostor_exp2, thresholds)

# EER
eer1_idx = np.argmin(np.abs(fmr1 - fnmr1))
eer2_idx = np.argmin(np.abs(fmr2 - fnmr2))
eer1 = (fmr1[eer1_idx] + fnmr1[eer1_idx]) / 2
eer2 = (fmr2[eer2_idx] + fnmr2[eer2_idx]) / 2
theta_eer1 = thresholds[eer1_idx]
theta_eer2 = thresholds[eer2_idx]

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fmr1*100, fnmr1*100, 'b-',  linewidth=2.5,
        label=f'Exp 1 — Real Data (EER = {eer1*100:.1f}%)')
ax.plot(fmr2*100, fnmr2*100, 'r--', linewidth=2.5,
        label=f'Exp 2 — Synthetic Data (EER = {eer2*100:.1f}%)')
ax.plot([0,100],[0,100],'k:',alpha=0.3, linewidth=1)
ax.scatter([eer1*100],[eer1*100], color='blue', zorder=5, s=80)
ax.scatter([eer2*100],[eer2*100], color='red',  zorder=5, s=80)
ax.set_xlabel('FMR — False Match Rate (%)', fontsize=12)
ax.set_ylabel('FNMR — False Non-Match Rate (%)', fontsize=12)
ax.set_title('Detection Error Tradeoff (DET) Curve\nReal vs. Synthetic Enrollment',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
plt.tight_layout()
plt.savefig('results/det_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print(f"  DET CURVE RESULTS")
print(f"{'='*50}")
print(f"  Exp 1 (Real)      EER = {eer1*100:.1f}%  at theta = {theta_eer1:.2f}")
print(f"  Exp 2 (Synthetic) EER = {eer2*100:.1f}%  at theta = {theta_eer2:.2f}")
print(f"{'='*50}")
print(f"  DET curve saved -> results/det_curve.png")
