import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def get_empirical_g_and_alpha(csv_file):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Extract the relevant columns
    explore_steps = data['median exp steps']
    exploit_steps = data['median scav steps']

    # Perform PCA on the data
    x = explore_steps
    y = exploit_steps
    pca = PCA(n_components=2)
    score = pca.fit_transform(np.column_stack((x, y)))

    # Ensure the empirical g (PC 1) decreases with the number of steps
    corrval = np.corrcoef(score[:, 0], explore_steps)[0, 1]
    if corrval > 0:
        pca.components_[0, :] = -pca.components_[0, :]
        score[:, 0] = -score[:, 0]

    # Ensure the empirical alpha (PC 2) increases with a tendency towards exploit
    corrval = np.corrcoef(score[:, 1], exploit_steps - explore_steps)[0, 1]
    if corrval < 0:
        pca.components_[1, :] = -pca.components_[1, :]
        score[:, 1] = -score[:, 1]

    # Extract g and alpha
    g = score[:, 0]
    alpha = score[:, 1]

    # Create dictionaries with subjects as keys. If the subject ID has a space in it, but the string immediately before the sapce
    # Create the subject keys, cutting string after space
    subjects_keys = [subject_key.split()[0] for subject_key in data['ID']]
    # Use the subjects_keys as keys and the g values as values:

    g_dict = dict(zip(subjects_keys, g))
    alpha_dict = dict(zip(subjects_keys, alpha))

    #g_dict = {f'sub-{int(subject_id):03d}': g[idx] for idx, subject_id in enumerate(data['ID'])}
    #alpha_dict = {f'sub-{int(subject_id):03d}': alpha[idx] for idx, subject_id in enumerate(data['ID'])}

    # Sort dictionaries by keys
    g_dict = dict(sorted(g_dict.items()))
    alpha_dict = dict(sorted(alpha_dict.items()))

    return g_dict, alpha_dict