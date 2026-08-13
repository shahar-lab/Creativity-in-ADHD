import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv("Data/task/cfg_all_clean.csv")

pca_data = data[["median exp steps", "median scav steps"]].copy()
valid_rows = pca_data.dropna().index

explore_steps = data.loc[valid_rows, "median exp steps"]
exploit_steps = data.loc[valid_rows, "median scav steps"]

pca = PCA(n_components=2)
score = pca.fit_transform(np.column_stack((explore_steps, exploit_steps)))

corrval = np.corrcoef(score[:, 0], explore_steps)[0, 1]
if corrval > 0:
    score[:, 0] = -score[:, 0]

corrval = np.corrcoef(score[:, 1], exploit_steps - explore_steps)[0, 1]
if corrval < 0:
    score[:, 1] = -score[:, 1]

data["g_empirical"] = np.nan
data["alpha_empirical"] = np.nan

data.loc[valid_rows, "g_empirical"] = score[:, 0]
data.loc[valid_rows, "alpha_empirical"] = score[:, 1]

data.to_csv("Data/task/cfg_all_with_g_alpha.csv", index=False)
print("saved")
