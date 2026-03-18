# Setting up model and training loop

## Model architecture

For this classification task we choose the EfficientNetV2-S model, which is a convolutional neural network architecture that has been shown to achieve state-of-the-art performance on image classification tasks while being computationally efficient. We will use the implementation of EfficientNetV2-S provided by `torchvision`. We will also use `torchvision` to load the ImageNet pretrained model weights.

```python
# models/efficientnet.py

class EfficientNet_V2_S(nn.Module):
    def __init__(self,
                 num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            )
        # modify model for number of classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280,
                       out_features=num_classes,
                       bias=True)
        )
    
    def forward(self, x):
        output = self.model(x)
        return output
```

## Setting up training loop

Here, we first set up a pytorch Dataset class and define a dataloder that will be used in the training loop. 

```python
class PlantVillageDataset(Dataset):
    def __init__(...):
        train_path = Path("PlantVillage") / "train.txt"
        # read path to the images and corresponding labels
        self.images, self.labels = read_data(train_path)
        # define data augmentation transform
        self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])
    ...

    def __getitem__(self, idx):
        image_path, label = self.images[idx], self.labels[idx]
        image = read_8bit_image(image_path)
        image = self.transform(image)
        return image, label

def loader(dataset:Dataset,
           batch_size:int,
           num_workers:int=1,
           ):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=dataset.shuffle,
                        num_workers=num_workers)
    return loader
```

Once a loader is set up, we can define the training loop. We will use the Adam optimizer and CrossEntropyLoss for this classification task. We will also use a learning rate scheduler to adjust the learning rate during training.

```python
class PlantTrainer(BaseTrainer):
    # BaseTrainer defines the basic functionality for training and validation loops, logging, checkpointing, etc.
    def train_one_epoch(self, epoch):
        total_loss = 0

        self.model.train()

        for k, batch in enumerate(self.train_loader):
            images, labels = batch
            images = images.to(device=self.device, memory_format=torch.channels_last)
            labels = labels.to(self.device)

            with self.amp_autocast:
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
            
            self.optimizer.zero_grad()
            if self.loss_scaler is None:
                loss.backward()
                self.optimizer.step()
            else:
                self.loss_scaler.scale(loss).backward()
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()

            if k%10 == 0:
                self.logger.info(f"Epoch {epoch}/{self.config['num_epochs']}  Iter {k:05d} : {loss:0.4f}")
            total_loss += loss
            if k == 20:
                break
        return total_loss
```

