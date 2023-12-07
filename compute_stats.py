""" 
Compute image statistics on the training set, i.e. mean and standard deviation on the training set.
"""

import numpy as np
from util.dataset import CTData
from tqdm import tqdm
from torch.utils.data import DataLoader


db = CTData(path_to_osic_data_dir="/SAN/medic/IPF/segmentations/osic_mar_23_unpadded/",
            path_to_lsut_data_dir="/cluster/project0/LUST_SSL/lsut_resampled_z256256_lungonly_h5/",
            path_to_df="ct_data.csv",
            mode="train",
            num_slices=16,
            seed=0,
            store_data_to_tmpfs=True,
            mean=0.6894,
            std=0.3069,
            )
dl = DataLoader(db, batch_size=1, shuffle=False, num_workers=32)

mean = 0.0
std = 0.0
TOTAL_PIXELS = 0

for sample in tqdm(dl):
    seq = sample['sequence'][0,0]
    msk = sample['mask'][0,0].bool()

    seq = seq[msk]
    mean += seq.sum().item()
    TOTAL_PIXELS += seq.numel()
mean /= TOTAL_PIXELS

# standard deviation
for sample in tqdm(dl):
    seq = sample['sequence'][0,0]
    msk = sample['mask'][0,0].bool()
    seq = seq[msk]
    std += ((seq - mean) ** 2).sum().item()
std = np.sqrt(std / TOTAL_PIXELS)

print(f"mean: {mean}")
print(f"std: {std}")