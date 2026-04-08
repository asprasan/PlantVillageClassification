import numpy
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def _preprocess_images(images_paths: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_paths: a txt file with image paths
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    with open(images_paths, 'r') as f:
        images_classes = f.readlines()

    image_names = [image_path.strip().split(',')[0] for image_path in images_classes]
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []
    count = 0
    for image_name in batch_filenames:
        image_filepath = image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = numpy.float32(pillow_img) - numpy.array(
            [123.68, 116.78, 103.94], dtype=numpy.float32
        )
        input_data = input_data / 255
        input_data = input_data - numpy.array([0.485, 0.456, 0.406],
                                              dtype=numpy.float32)
        input_data = input_data / numpy.array([0.229, 0.224, 0.225],
                                              dtype=numpy.float32)
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
        count += 1
    print(f"preprocessed {count} images")
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class PlantVillageDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_txt: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_txt, height, width, size_limit=5
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

