import yaml
import argparse

from utils.utils import validate_path
from trainers import TRAINER_REGISTRY, plant_trainer

def parse_args():
    parser = argparse.ArgumentParser("arguments for PlantVillage classification")
    parser.add_argument("output_path", type=validate_path,
                        help="folder containing the trained checkpoint")
    return parser.parse_args()

def main(args):
    # load config
    config_path = args.output_path / 'config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['batch_size'] = 1
    trainer_class = TRAINER_REGISTRY[config['trainer']]
    trainer = trainer_class(config,
                            1
                            )
    test_loader = trainer.setup_dataloader('test')
    trainer.evaluate_onnx(test_loader)
    trainer.evaluate_torch(test_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
