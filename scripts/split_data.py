"""Split a txt data info file into train/val sets."""

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split dataset info file')
parser.add_argument('--info_file', required=True, type=str)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--cat_loc', type=int, default=1)
args = parser.parse_args()

with open(args.info_file, 'r') as f:
    all_path = [line.strip() for line in f.readlines()]

all_cat = np.unique([line.split('/')[args.cat_loc] for line in all_path])
print(f'{all_cat}\n{len(all_cat)} categories detected')

cat2path = {
    cat: [path for path in all_path if cat == path.split('/')[args.cat_loc]]
    for cat in all_cat
}
train_paths, val_paths = [], []
for cat, paths in cat2path.items():
    np.random.shuffle(paths)
    n_val = len(paths) * args.val_ratio
    if n_val <= 1:
        n_val = 1
    else:
        n_val = int(n_val)
    train_paths.extend(paths[n_val:])
    val_paths.extend(paths[:n_val])

print(f'Split {len(all_path)} data into {len(train_paths)} training '
      f'and {len(val_paths)} validation')

# save to {}.train.txt and {}.val.txt
with open(args.info_file.replace('.txt', '.train.txt'), 'w') as f:
    f.write('\n'.join(train_paths))
with open(args.info_file.replace('.txt', '.val.txt'), 'w') as f:
    f.write('\n'.join(val_paths))
