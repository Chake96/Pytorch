

class Dataset():
    def __init__(self, x_ds, y_ds):
        self.x_dataset = x_ds
        self.y_dataset = y_ds

        def __len__(self):
            return len(self.x_dataset)

        def __getitem__(self, i):
            return self.x_dataset[i], self.y_dataset[i]