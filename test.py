
from datasets.hc_dataset import HCDataset
from datasets.loading import load_data

dataset = 'reddit'
# create dataset
x, y_true, similarities = load_data(dataset)
#
# dataset = HCDataset(x, y_true, similarities, num_samples=args.num_samples)

print(type(x))
print(type(y_true))
print(type(similarities))
print(x.shape)
print(y_true.shape)
print(similarities.shape)