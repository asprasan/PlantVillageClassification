# Evaluation with ONNX and Pytorch

ONNX is a powerful tool for deploying machine learning models across different platforms and frameworks. In this section, we will discuss how to evaluate and deploy your ONNX models effectively.

## Exporting Models to ONNX and Pytorch formats

Pytorch provides a convenient way to export models to ONNX format. During training, we export our model to both ONNX and PyTorch formats. This allows us to evaluate the ONNX model using ONNX Runtime, which is optimized for performance.

```python
# trainers/plant_trainer.py
# export checkpoint.pth
model_dict = {'model': self.model.state_dict(),
                                "epoch": epoch,
                                "accuracy": accuracy,
                                "optimizer_state": self.optimizer.state_dict()
                                }
torch.save(model_dict, self.output_path / "checkpoint.pth")
# export onnx
example_inputs = (torch.randn(1, 3, 224, 224),)
onnx_model = torch.onnx.export(self.model,
                                example_inputs,
                                dynamo=True)
onnx_model.save(self.output_path / "checkpoint.onnx")
```

## Using pytorch for model inference

The `plant_trainer.py` script includes a method for evaluating the model using PyTorch. This method loads the saved checkpoint and evaluates the model on the test dataset.

### Results of evaluation on the test dataset

