# Data preparation

Thankfully, the PlantVillage dataset is a clean and well-structured dataset, which makes data preparation easier. One of the main tasks in data preparation is to split the dataset into training, validation, and test sets. This can be done using the `train_test_split` function from the `sklearn` library.

## Data splitting

In machine learning, it's important to split the dataset while maintaining the class distribution. This is known as stratified splitting. The `train_test_split` function allows us to perform stratified splitting by using the `stratify` parameter.

Here's how you can split the dataset into training, validation, and test sets:

```python
from sklearn.model_selection import train_test_split

test_size = args.test # test split size
val_size = args.val # validation split size
train_size = 1 - test_size - val_size
val_size = val_size * (train_size + val_size)
# First, split the dataset into training+validation and test sets
images_tv, images_test, labels_tv, labels_test = train_test_split(images,
                                                                labels,
                                                                test_size=args.test,
                                                                random_state=args.seed,
                                                                stratify=labels
                                                                )

# Now split the training+validation set into training and validation sets
images_train, images_val, labels_train, labels_val = train_test_split(images_tv,
                                                                    labels_tv,
                                                                    test_size=val_size,
                                                                    random_state=args.seed,
                                                                    stratify=labels_tv
                                                                    )
```

The `train_test_split` function is called twice. The first call splits the dataset into a combined training+validation set and a test set. The second call then splits the combined training+validation set into separate training and validation sets. By using the `stratify` parameter, we ensure that the class distribution is maintained in each of the splits.

The split data can be saved as txt files (or csv files) for later use. Each line in the txt file can contain the image path and its corresponding label, separated by a comma. This way, you can easily load the data during training and evaluation.

```python
import os
output_dir = "PlantVillage"
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for img_path, label in zip(images_train, labels_train):
        f.write(f"{img_path},{label}\n")
with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for img_path, label in zip(images_val, labels_val):
        f.write(f"{img_path},{label}\n")
with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
    for img_path, label in zip(images_test, labels_test):
        f.write(f"{img_path},{label}\n")
```

Now we shall also track the train, val, and test splits using `dvc` to ensure that we can reproduce the splits in the future. This is important for maintaining the integrity of our experiments and ensuring that we can compare results across different runs.

```bash
dvc add PlantVillage/train.txt
dvc add PlantVillage/val.txt
dvc add PlantVillage/test.txt
git commit -m "Add train, val, and test splits for PlantVillage dataset"
git push
dvc push
```

By following these steps, we have successfully prepared the PlantVillage dataset for training our machine learning models. We have split the dataset into training, validation, and test sets while maintaining the class distribution, and we have also tracked these splits using `dvc` for reproducibility.