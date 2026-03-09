from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import read_8bit_image

class PlantDataset(Dataset):

    def __init__(self,
                 root_path:str|Path,
                 split:str):
        super().__init__()
        assert split in ["train", "val", "test"]
        
        self.shuffle = not split == "test"
        if isinstance(root_path, str):
            root_path = Path(root_path)
        data_txt_path = root_path / f"{split}.txt"
        self.images, self.labels = self.read_data(data_txt_path)
        self.transform = self.get_transform(split)

    def get_transform(self, split):
        if split == "test":
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                transforms.CenterCrop(size=224)
            ])
        else:
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ])
        return transform

    def read_data(self, txt_path):
        with open(txt_path, "r") as f:
            lines = f.readlines()
        images = []
        labels = []
        for line in lines:
            image, label = line.strip().split(',')
            images.append(Path(image))
            labels.append(int(label))
        print(f"found {len(images)} images")
        return images, labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx], self.labels[idx]
        image = read_8bit_image(image_path)
        image = self.transform(image)
        return image, label
    
if __name__ == "__main__":
    dataset = PlantDataset(Path("PlantVillage"), "test")
    loader = DataLoader(dataset, shuffle=dataset.shuffle)
    loader = iter(loader)
    image, label = next(loader)
    print(image.shape)
    print(label)
