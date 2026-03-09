import yaml
import argparse
# import wandb
import torch

from utils import validate_path, get_logger
from models import MODEL_REGISTRY, efficientnet
from data import PlantDataset
from loader import loader

def parse_args():
    parser = argparse.ArgumentParser("arguments for PlantVillage classification")
    parser.add_argument("config_path", type=validate_path,
                        help="yml file to load the config paramters from")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from last checkpoint")
    parser.add_argument("--checkpoint", type=validate_path,
                        help="checkpoint to resume from")
    return parser.parse_args()

def main(args):
    # load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger = get_logger("train")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = MODEL_REGISTRY[config["model"]]
    # create dataset and dataloader
    model = model_class(config["num_classes"])
    model.to(device)
    # create dataset and dataloader
    train_dataset = PlantDataset('PlantVillage',
                                 'train')
    val_dataset = PlantDataset('PlantVillage',
                                 'val')
    
    train_loader = loader(train_dataset,
                          config["batch_size"],
                          )
    logger.info(f"created train loader with length {len(train_loader)}")
    val_loader = loader(val_dataset,
                        config["batch_size"]
                        )
    logger.info(f"created val loader with length {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        # training loop
        model.train()
        for k, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            loss = loss_fn(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (k+1)%10 == 0:
                logger.info(f"Loss at epoch {epoch}/{config['num_epochs']}: {loss:0.4f}")
                break
        
        # validation loop
        logger.info(f"Validating at epoch {epoch}")
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            for k, batch in enumerate(val_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                pred = model(images)
                # convert pred to labels and compute accuracy
                _, pred_labels = torch.max(pred, 1)
                total += labels.size(0)
                correct += (pred_labels == labels).sum().item()
            logger.info(f"Accuracy at epoch {epoch} = {correct*100/total}%")
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
