from torch.utils.data import Dataset,DataLoader

def loader(dataset:Dataset,
           batch_size:int,
           num_workers:int=1,
           ):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=dataset.shuffle,
                        num_workers=num_workers)
    return loader