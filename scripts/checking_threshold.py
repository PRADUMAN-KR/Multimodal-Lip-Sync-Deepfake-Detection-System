from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

pred_path = Path("results_zarr") / "predictions.csv"
df = pd.read_csv(pred_path)

y_true = df["ground_truth"].to_numpy()   # 0=real, 1=fake
p_real = df["confidence"].to_numpy()     # P(real)

for t in [0.3, 0.4, 0.5, 0.6, 0.7,0.8]:
    # Predict fake (1) when P(real) < t, real (0) otherwise
    y_pred = (p_real < t).astype(int)    # 0=real, 1=fake
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"threshold={t:.2f}  TN={tn} FP={fp} FN={fn} TP={tp}")