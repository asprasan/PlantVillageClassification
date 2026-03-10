from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import subprocess
from typing import Dict, Any
from contextlib import suppress
import torch
from utils import get_logger

class BaseTrainer(ABC):

    def __init__(self, config: Dict[str, Any],
                 workers:int,
                 resume:bool=False,
                 resume_checkpoint:Path=None):
        self._config = config
        self.workers = workers
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self.current_epoch = 0
        self.logger = get_logger("train")
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        self.model = self.setup_model()
        self.train_loader = self.setup_dataloader(split='train')
        self.val_loader = self.setup_dataloader(split='val')
        self.optimizer = self.setup_optimizer()
        self.loss_fn = self.setup_loss_fn()
        self.train = None
        self.autocast = self.config['amp']
        self.setup_autocast()

        self.output_path = Path('results') / self.config["experiment"]
        if self.output_path.exists():
            raise ValueError("output directory already exists")
        self.output_path.mkdir()
        self.log_experiment_details()

    def log_experiment_details(self):
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        with open(self.output_path / "git_hash.txt", 'w') as f:
            f.write(f"{git_hash}")
        
        with open(self.output_path / "config.yml", 'w') as f:
            yaml.dump(self.config, f)

    @property
    def config(self):
        return self._config

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def setup_dataloader(self, split: str):
        pass

    def setup_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config["lr"],
                                      weight_decay=0.01
                                      )
        return optimizer

    def setup_loss_fn(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn
    
    def setup_autocast(self):
        self.loss_scaler = None
        self.amp_autocast = suppress()
        if self.autocast:
            self.loss_scaler = torch.amp.GradScaler("cpu")
            self.amp_autocast = torch.autocast(device_type="cpu",
                                               dtype=torch.bfloat16)
    
    def load_model(self):
        checkpoint_dict = torch.load(self.resume_checkpoint,
                                     map_location='cpu')
        checkpoint = checkpoint_dict['model']
        self.model.load_state_dict(checkpoint)
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state'])
        self.current_epoch = checkpoint_dict['epoch']

