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

@register_trainer("plant_qat_trainer")
class PlantQATTrainer(BaseTrainer):

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
        quantization_config = torch.quantization.get_default_qat_qconfig("fbgemm")
        model.qconfig = quantization_config
        model.qconfig = torch.ao.quantization.default_qconfig
        torch.quantization.prepare_qat(model, inplace=True)
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
            if k == 50:
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
        self.logger.info("Evaluating with ONNXRuntime")
        total = 0
        latencies = []
        all_gt_labels = []
        all_pred_labels = []
        ort_session = onnxruntime.InferenceSession(
            self.output_path / "checkpoint_int8.onnx",
            providers=["CPUExecutionProvider"]
        )
        for k, batch in enumerate(loader):
            tensor, labels = batch
            onnx_input = [tensor.numpy(force=True)]


            onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_input)}
            # ONNX Runtime returns a list of outputs
            start_time = time.time()
            onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
            latencies.append(time.time() - start_time)
            pred_labels = np.argmax(onnxruntime_outputs, 1)
            all_gt_labels.extend(labels.data.cpu().numpy().flatten().tolist())
            all_pred_labels.extend(pred_labels.flatten().tolist())
            total += labels.size(0)
        confusion_mat = self.compute_stats(all_pred_labels, all_gt_labels)
        self.logger.info(f"Inference speed: {total/sum(latencies):0.2f} frames/second")


    def evaluate_torch(self, loader):
        # load the model
        self.logger.info("Evaluating with Pytorch")
        self.resume_checkpoint = self.output_path / "checkpoint.pth"
        self.load_model()
        with torch.no_grad():
            self.model.eval()
            total = 0
            latencies = []
            all_gt_labels = []
            all_pred_labels = []
            for k, batch in enumerate(loader):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                start_time = time.time()
                pred = self.model(images)
                latencies.append(time.time() - start_time)
                # convert pred to labels and compute accuracy
                pred = torch.softmax(pred, 1)
                _, pred_labels = torch.max(pred, 1)
                all_gt_labels.extend(labels.data.cpu().numpy().flatten().tolist())
                all_pred_labels.extend(pred_labels.flatten().tolist())
                total += labels.size(0)

        confusion_mat = self.compute_stats(all_pred_labels, all_gt_labels)
        self.logger.info(f"Inference speed: {total/sum(latencies):0.2f} frames/second")
    
    def compute_stats(self, pred_labels, gt_labels):
        confusion_mat = np.zeros((2,2))
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            confusion_mat[gt_label, pred_label] += 1
        self.logger.info("Confusion matrix: rows indicate GT labels")
        self.logger.info(f"{confusion_mat}")
        accuracy = (confusion_mat[0,0] + confusion_mat[1,1])/ np.sum(confusion_mat)
        self.logger.info(f"Accuracy= {accuracy*100:0.4f}")
        return confusion_mat