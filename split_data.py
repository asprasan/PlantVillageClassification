import argparse

from sklearn.model_selection import train_test_split

from utils import validate_path, get_logger

IMAGE_EXTS = ["jpg"]

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=validate_path, help="path to the dataset")
    parser.add_argument("--val", type=float, default=0.1, help="percentage of val data")
    parser.add_argument("--test", type=float, default=0.1, help="percentage of val data")
    parser.add_argument("--seed", type=int, default=42, help="percentage of val data")

    args = parser.parse_args()
    return args

def write_to_txt(root_path, images, labels, name="train"):
    with open(root_path / f"{name}.txt", "w") as f:
        for image, label in zip(images, labels):
            f.write(f"{str(image)},{label}\n")

def split_data(args):
    """given a data folder containing sub-folders for each class
    the function generates 3 text files 

    Parameters
    ----------
    args : _type_
        _description_
    """
    class_folders = sorted(args.root_path.glob("*"))
    test_size = args.test
    val_size = args.val
    train_size = 1 - test_size - val_size
    val_size = val_size * (train_size + val_size)

    logger = get_logger("split_data")
    logger.info(f"found {len(class_folders)} image folders")
    logger.debug(f"{class_folders=}")

    images = []
    labels = []
    label = 0
    for class_folder in class_folders:
        if class_folder.is_dir():
            # list all images
            curr_images = []
            for image_ext in IMAGE_EXTS:
                curr_images.extend(sorted(class_folder.glob(f"*.{image_ext}")))
            logger.debug(f"found {len(curr_images)} images for label {label}")
            curr_labels = [label,]*len(curr_images)
            images.extend(curr_images)
            labels.extend(curr_labels)
            label += 1
    
    # split data
    logger.info("split train-val and test data")
    images_tv, images_test, labels_tv, labels_test = train_test_split(images,
                                                                    labels,
                                                                    test_size=args.test,
                                                                    random_state=args.seed,
                                                                    stratify=labels)
    logger.debug(f"number of test images = {len(labels_test)}")
    images_train, images_val, labels_train, labels_val = train_test_split(images_tv,
                                                                          labels_tv,
                                                                          test_size=val_size,
                                                                          random_state=args.seed,
                                                                          stratify=labels_tv)
    logger.debug(f"number of train images = {len(labels_train)}")
    logger.debug(f"number of val images = {len(labels_val)}")
    
    # write the output image paths and class labels into txt files
    write_to_txt(args.root_path,
                 images_train,
                 labels_train,
                 "train")
    write_to_txt(args.root_path,
                 images_val,
                 labels_val,
                 "val")
    write_to_txt(args.root_path,
                 images_test,
                 labels_test,
                 "test")

if __name__ == "__main__":
    args = parser_args()
    split_data(args)
