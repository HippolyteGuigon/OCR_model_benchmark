import torch
import warnings

from PIL import Image
from typing import Union, List

from transformers import LayoutLMv2Processor, LayoutLMTokenizer
from transformers import DonutProcessor
from ocr_benchmark.utils.data_loading import load_data

warnings.filterwarnings("ignore")

tokenizer = LayoutLMTokenizer.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)


def align_labels_with_tokens(labels, word_ids):
    """
    Align the word-level labels with the token-level
    labels after tokenization. Each word might be split
    into multiple tokens, and we need to duplicate the
    labels accordingly.

    Arguments:
        - labels: List[int]: The original word-level labels.
        - word_ids: The list of word_ids from the tokenizer,
        indicating which word each token corresponds to.

    Returns:
        - aligned_labels: List[int]: The token-level labels
        aligned with the tokens after tokenization.
    """
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            aligned_labels.append(labels[word_id])
            previous_word_id = word_id
        else:
            aligned_labels.append(-100)

    return aligned_labels


def image_preprocessing(
    image_index: int, **kwargs
) -> Union[LayoutLMv2Processor, List[str]]:
    """
    The purpose of this function is to preprocess an image from the dataset
    and return the necessary encoding and related image information
    for the LayoutLMv2 model.
    """
    if "dataset" not in kwargs:
        dataset = load_data()
    if "processor" not in kwargs:
        processor = LayoutLMv2Processor.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", apply_ocr=False
        )

    image_path = dataset["test"][image_index]["image_path"]
    annotations = dataset["test"][image_index]["words"]
    boxes = dataset["test"][image_index]["bboxes"]
    labels = dataset["test"][image_index]["ner_tags"]

    image_info = [image_path, annotations, boxes, labels]

    image = Image.open(image_path)
    image = image.convert("RGB")

    encoding = processor(image, annotations, boxes=boxes, return_tensors="pt")

    word_ids = encoding.word_ids()
    aligned_labels = align_labels_with_tokens(labels, word_ids)

    encoding["labels"] = torch.tensor(aligned_labels)

    return encoding, image_info


def image_preprocessing_donut(image_path: str):
    """
    Préparer une image pour Donut.
    """

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    return pixel_values


processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", apply_ocr=False
)


def preprocess_funsd(example):
    image_path = example["image_path"]
    image = Image.open(image_path).convert("RGB")
    words = example["words"]
    boxes = example["bboxes"]
    labels = example["ner_tags"]

    # Appliquer la tokenisation et l'encodage LayoutLMv2
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # Inclure l'image dans l'encodage avec la clé 'image'
    encoding["image"] = processor.feature_extractor(
        images=image, return_tensors="pt"
    )["pixel_values"]

    # Aligner les labels sur les tokens
    word_ids = encoding.word_ids()
    aligned_labels = align_labels_with_tokens(labels, word_ids)
    encoding["labels"] = torch.tensor(aligned_labels)

    return {
        key: val.squeeze(0) for key, val in encoding.items()
    }  # Squeeze pour obtenir des dimensions correctes
