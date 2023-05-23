import pickle
import numpy as np

with open('./all_data/rotations.pkl', "rb") as f:
    rotation_idx = pickle.load(f)
    rotations = np.random.binomial(1, 0.5, len(rotation_idx))
    rotation_idx = np.where(rotations)
    print(rotation_idx)
    with open('./all_data/rotations.pkl', 'wb') as f:
        pickle.dump(rotation_idx, f)
with open('./all_data/label_projection.pkl', "rb") as f:
    labels = pickle.load(f)
    print(labels)