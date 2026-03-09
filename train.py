import yaml
import argparse
import wandb
import torch

from utils import validate_path, get_logger
from models import MODEL_REGISTRY, efficientnet
from data import PlantDataset
from loader import loader

def parse_args():
    parser = argparse.ArgumentParser("arguments for PlantVillage classification")
    parser.add_argument("config_path", type=validate_path,
                        help="yml file to load the config paramters from")
    parser.add_argument("--workers", type=int, default=2,
                        help="number of dataloader workers")
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
                          num_workers=args.workers)
    logger.info(f"created train loader with length {len(train_loader)}")
    val_loader = loader(val_dataset,
                        config["batch_size"],
                        num_workers=args.workers)
    logger.info(f"created val loader with length {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_scaler = torch.amp.GradScaler("cpu")
    best_acc = 0.0
    with wandb.init(
        project="plantvillage-classification",
        notes="Bell-pepper healthy vs diseased",
        config=config
    ) as run:
        for epoch in range(config["num_epochs"]):
            # training loop
            total_loss = 0
            run.config["optimizer"] = "AdamW"
            run.config["amp"] = True

            model.train()
            for k, batch in enumerate(train_loader):
                images, labels = batch
                images = images.to(device=device, memory_format=torch.channels_last)
                labels = labels.to(device)

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    pred = model(images)
                    loss = loss_fn(pred, labels)
                
                optimizer.zero_grad()
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()

                if k%10 == 0:
                    logger.info(f"Epoch {epoch}/{config['num_epochs']}  Iter {k:05d} : {loss:0.4f}")
                total_loss += loss
            run.log({"epoch": epoch,
                     "loss": total_loss/k})

            # validation loop
            logger.info(f"Validating at epoch {epoch}")
            with torch.no_grad():
                model.eval()
                total = 0
                correct = 0
                for k, batch in enumerate(val_loader):
                    images, labels = batch
                    images = images.to(device, memory_format=torch.channels_last)
                    labels = labels.to(device)

                    pred = model(images)
                    # convert pred to labels and compute accuracy
                    _, pred_labels = torch.max(pred, 1)
                    total += labels.size(0)
                    correct += (pred_labels == labels).sum().item()
                accuracy = correct / total
                logger.info(f"Accuracy at epoch {epoch} = {accuracy*100}%")
                run.log({"epoch": epoch,
                         "accuracy": accuracy*100})
                if best_acc <= accuracy:
                    # save the model with the best accuracy
                    logger.info(f"Saving model at epoch {epoch} with best accuracy {accuracy}")
                    model_dict = {'model': model.state_dict(),
                                "epoch": epoch,
                                "accuracy": accuracy,
                                "optimizer": optimizer.state_dict()
                                }
                    torch.save(model_dict, "results/checkpoint.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
