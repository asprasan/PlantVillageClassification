import yaml
import argparse

from utils import validate_path
from trainers import TRAINER_REGISTRY, plant_trainer

def parse_args():
    parser = argparse.ArgumentParser("arguments for PlantVillage classification")
    parser.add_argument("config_path", type=validate_path,
                        help="yml file to load the config paramters from")
    parser.add_argument("--workers", type=int, default=2,
                        help="number of dataloader workers")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from last checkpoint")
    parser.add_argument("--resume_checkpoint", type=validate_path,
                        help="checkpoint to resume from")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="disable wandb logging for this run")
    return parser.parse_args()

def main(args):
    # load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    trainer_class = TRAINER_REGISTRY[config['trainer']]
    breakpoint()
    trainer = trainer_class(config,
                            args.workers,
                            resume=args.resume,
                            resume_checkpoint=args.resume_checkpoint
                            )
    trainer.training()

if __name__ == "__main__":
    args = parse_args()
    if args.disable_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"
    main(args)
