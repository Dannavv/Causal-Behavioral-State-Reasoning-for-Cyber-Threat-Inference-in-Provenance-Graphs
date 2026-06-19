import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv('data/processed/streamspot_edges.csv', index_col='timestamp')
labels = df['is_attack'].values
ts_df = df.drop(columns=['is_attack'])

benign_mask = labels == 0
attack_mask = labels == 1
benign_indices = np.where(benign_mask)[0]
attack_indices_arr = np.where(attack_mask)[0]

n_benign_train = int(0.6 * len(benign_indices))
train_benign_idx = benign_indices[:n_benign_train]
test_benign_idx = benign_indices[n_benign_train:]

# Test set = remaining 40% benign + ALL attack windows
test_idx = np.sort(np.concatenate([test_benign_idx, attack_indices_arr]))

train_df_ss = ts_df.iloc[train_benign_idx]
test_df_ss = ts_df.iloc[test_idx]

train_labels = labels[train_benign_idx]
test_labels_ss = labels[test_idx]

split_idx = len(train_df_ss)
print('train size:', split_idx, 'sum train labels:', sum(train_labels))
print('test size:', len(test_df_ss), 'sum test labels:', sum(test_labels_ss))
