# Experiment tracking with Wandb

Weights & Biases (Wandb) is a popular tool for experiment tracking and visualization in machine learning. It allows you to log your training metrics, visualize them in real-time, and compare different runs of your experiments.

To use Wandb in your machine learning projects, you can follow these steps:

```bash
# install wandb
python -m pip install wandb
```

Next, you can initialize Wandb in your training script and log your metrics. Here's an example of how to do this:

```python
import wandb

def train(self):
    # Initialize a new run
    with wandb.init(
            project="plantvillage-classification",
            notes="Bell-pepper healthy vs diseased",
            config=self.config
            ) as run:
            for epoch in range(self.current_epoch,
                               self.config["num_epochs"]):
                # training loop
                run.config["optimizer"] = "AdamW"
                run.config["amp"] = self.autocast
                
                total_loss = self.train_one_epoch(epoch)
                run.log({"epoch": epoch,
                        "loss": total_loss/len(self.train_loader)})
```

`self.config` is a dictionary that contains the hyperparameters and other configuration settings for your training. You can log any metrics you want by using `run.log()`, which takes a dictionary of metric names and their corresponding values.

```python
# sample self.config dictionary:
self.config = {
            'experiment': 'baseline',
            'batch_size': 8,
            'lr': 0.0001,
            'num_epochs': 2,
            'model': 'efficientnet_v2_s',
            'trainer': 'plant_trainer',
            'data_path': 'PlantVillage',
            'num_classes': 2,
            'amp': True,
            'export_onnx': True
        }
```

By logging your metrics with Wandb, you can easily visualize them in the Wandb dashboard and compare different runs of your experiments. This can help you identify which hyperparameters and configurations lead to better performance.
