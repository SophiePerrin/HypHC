
from datasets.hc_dataset import HCDataset
from datasets.loading import load_data

dataset = 'reddit'
# create dataset
x, y_true, similarities = load_data(dataset)
#
# dataset = HCDataset(x, y_true, similarities, num_samples=args.num_samples)
