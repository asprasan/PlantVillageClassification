import time
import numpy as np
import torch
import wandb
from models import MODEL_REGISTRY, efficientnet
import onnxruntime

from trainers import register_trainer
from trainers.base_trainer import BaseTrainer
from data.dataset import PlantDataset
from data.loader import loader

@register_trainer("plant_trainer")
class PlantTrainer(BaseTrainer):

    def __init__(self,
                 config,
                 workers,
                 resume=False,
                 resume_checkpoint=None):
        super().__init__(config,
                         workers,
                         resume=resume,
                         resume_checkpoint=resume_checkpoint)

    def setup_model(self):
        model_class = MODEL_REGISTRY[self.config["model"]]
        # create dataset and dataloader
        model = model_class(self.config["num_classes"])
        return model

    def setup_dataloader(self, split):
        dataset = PlantDataset(self.config['data_path'],
                               split)
        return loader(dataset,
                      self.config["batch_size"],
                      num_workers=self.workers)
    
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
    
    def validate(self)->float:
        with torch.no_grad():
            self.model.eval()
            total = 0
            correct = 0
            for k, batch in enumerate(self.val_loader):
                images, labels = batch
                images = images.to(self.device, memory_format=torch.channels_last)
                labels = labels.to(self.device)

                pred = self.model(images)
                # convert pred to labels and compute accuracy
                pred = torch.softmax(pred, 1)
                _, pred_labels = torch.max(pred, 1)
                total += labels.size(0)
                correct += (pred_labels == labels).sum().item()
            accuracy = correct / total
        return accuracy
    
    def export_onnx(self):
        example_inputs = (torch.randn(1, 3, 224, 224),)
        onnx_model = torch.onnx.export(self.model,
                                       example_inputs,
                                       dynamo=True)
        onnx_model.save(self.output_path / "checkpoint.onnx")
        

    def training(self):
        best_acc = 0.0
        if self.resume:
            self.load_model()
            best_acc = torch.load(self.resume_checkpoint,
                                  map_location='cpu')['accuracy']
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
                        "loss": total_loss/len(self.val_loader)})

                # validation loop
                
                self.logger.info(f"Validating at epoch {epoch}")

                accuracy = self.validate()
                self.logger.info(f"Accuracy at epoch {epoch} = {accuracy*100}%")
                
                run.log({"epoch": epoch,
                        "accuracy": accuracy*100})
                if best_acc <= accuracy:
                    # save the model with the best accuracy
                    self.logger.info(f"Saving model at epoch {epoch} with best accuracy {accuracy}")
                    model_dict = {'model': self.model.state_dict(),
                                "epoch": epoch,
                                "accuracy": accuracy,
                                "optimizer_state": self.optimizer.state_dict()
                                }
                    torch.save(model_dict, self.output_path / "checkpoint.pth")
                    best_acc = accuracy
                    if self.config['export_onnx']:
                        self.export_onnx()

    def evaluate_onnx(self, loader):
        total = 0
        correct = 0
        start_time = time.time()
        for k, batch in enumerate(loader):
            tensor, label = batch
            onnx_input = [tensor.numpy(force=True)]

            ort_session = onnxruntime.InferenceSession(
                self.output_path / "checkpoint.onnx",
                providers=["CPUExecutionProvider"]
            )

            onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}

            # ONNX Runtime returns a list of outputs
            onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
            pred_labels = np.argmax(onnxruntime_outputs, 1)

            total += 1
            correct += label.item() == pred_labels[0]
        end_time = time.time() - start_time
        accuracy = correct / total
        self.logger.info(f"Accuracy with ONNX = {accuracy*100:0.4f}")
        self.logger.info(f"Inference speed: {total/end_time} frames/second")


    def evaluate_torch(self, loader):
        # load the model
        self.resume_checkpoint = self.output_path / "checkpoint.pth"
        self.load_model()
        with torch.no_grad():
            self.model.eval()
            total = 0
            correct = 0
            start_time = time.time()
            for k, batch in enumerate(loader):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(images)
                # convert pred to labels and compute accuracy
                pred = torch.softmax(pred, 1)
                _, pred_labels = torch.max(pred, 1)
                total += labels.size(0)
                correct += (pred_labels == labels).sum().item()
            end_time = time.time() - start_time
            accuracy = correct / total

        self.logger.info(f"Accuracy with Pytorch = {accuracy*100:0.4f}")
        self.logger.info(f"Inference speed: {total/end_time} frames/second")
