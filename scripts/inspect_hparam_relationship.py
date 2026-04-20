import os
import numpy as np
import pandas as pd
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

DIR = "sweep_results"
journal_file = os.path.join(DIR, "journal_deepmoe.log")

storage = JournalStorage(JournalFileStorage(journal_file))

study = optuna.create_study(
        study_name="deepmoe",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

# Assuming 'study' is your loaded Optuna study object
df = study.trials_dataframe()

# Extract just the two hyperparameters
# Note: Optuna prepends 'params_' to hyperparameter columns in the dataframe
hparams = df[['params_lambda_g', 'params_lr_moe_mul']].dropna()

# 1. Calculate Spearman Correlation (No transformation needed)
spearman_corr = hparams.corr(method='spearman')
print("Spearman Correlation:\n", spearman_corr)

# 2. Calculate Pearson Correlation (Log transformation required)
# We take the log10 to match your chart's axes
log_hparams = np.log10(hparams)
pearson_corr = log_hparams.corr(method='pearson')
print("\nPearson Correlation (Log-Log Space):\n", pearson_corr)