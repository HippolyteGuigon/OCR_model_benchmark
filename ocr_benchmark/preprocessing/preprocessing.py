import warnings
warnings.filterwarnings("ignore")

from PIL import Image

from transformers import LayoutLMv2Processor
from ocr_benchmark.utils.data_loading import load_data


def image_preprocessing(image_index: int, **kwargs):

    if "dataset" not in kwargs.items():
        dataset=load_data()
    if "processor" not in kwargs.items():
        processor=LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)

    image_path = dataset['test'][5]['image_path']
    annotations = dataset['test'][5]['words']
    boxes = dataset['test'][5]['bboxes']  
    labels = dataset['test'][5]['ner_tags']

    image = Image.open(image_path)
    image = image.convert("RGB")

    encoding = processor(image, annotations, boxes=boxes, return_tensors="pt")
    
    return encoding