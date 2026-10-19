from pathlib import Path
import pandas as pd
from stroke_rf_timeline.pipeline import train_validate, RFConfig

df = pd.read_csv(Path(__file__).parent / "template_clinical_data.csv")
gs, metrics, preds = train_validate(df, config=RFConfig())
print(metrics)
print(preds.head())
print("Best params:", gs.best_params_)
