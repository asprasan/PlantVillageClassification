from pathlib import Path
import argparse
import numpy as np
import torch
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationMethod

from data.quantize_loader import PlantVillageDataReader
from models.efficientnet import EfficientNet_V2_S

def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="PlantVillage/val.txt", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = EfficientNet_V2_S(2)
    checkpoint_path = Path(args.input_model).parent / "checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model,
                    example_input,
                    args.input_model,
                    input_names=["input"],
                    output_names=["output"])
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    dr = PlantVillageDataReader(
        calibration_dataset_path, input_model_path
    )

    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        calibrate_method=CalibrationMethod.Percentile,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()

