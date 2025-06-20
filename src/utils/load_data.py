import json


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_dataset(config):
    train_dataset = load_jsonl(config.train_path)
    dev_dataset = load_jsonl(config.dev_path)
    return train_dataset, dev_dataset


class Dataset_Iterater(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            iter_dataset = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            return iter_dataset

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            iter_dataset = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return iter_dataset

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = Dataset_Iterater(dataset, config.batch_size)
    return iter
