import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from typing import Union, List 

from transformers import LayoutLMv2Processor
from ocr_benchmark.utils.data_loading import load_data


def image_preprocessing(image_index: int, **kwargs) -> Union[LayoutLMv2Processor, List[str]]:
    """
    The purpose of this function is to preprocess an image from the dataset
    and return the necessary encoding and related image information
    for the LayoutLMv2 model.

    Arguments:
        - image_index: int: The index of the image in the dataset to process.
        - **kwargs: optional keyword arguments:
            - dataset: a preloaded dataset (if not provided, it loads via load_data()).
            - processor: a preloaded LayoutLMv2Processor (if not provided, it uses a pre-trained processor).

    Returns:
        - encoding: LayoutLMv2Processor: The encoded representation of the image and associated annotations, including bounding boxes.
        - image_info: List[str]: A list containing the image path, annotations, bounding boxes, and labels.

    Notes:
        - If a dataset or processor is not provided via `kwargs`, they will be initialized within the function.
        - The image is automatically converted to RGB format for compatibility with LayoutLMv2.
    """

    if "dataset" not in kwargs.items():
        dataset=load_data()
    if "processor" not in kwargs.items():
        processor=LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

    image_path = dataset['test'][5]['image_path']
    annotations = dataset['test'][5]['words']
    boxes = dataset['test'][5]['bboxes']  
    labels = dataset['test'][5]['ner_tags']

    image_info=[image_path, annotations, boxes, labels]

    image = Image.open(image_path)
    image = image.convert("RGB")

    encoding = processor(image, annotations, boxes=boxes, return_tensors="pt")
    
    return encoding, image_info