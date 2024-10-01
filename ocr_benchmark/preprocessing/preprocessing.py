import pytesseract
import torch
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Union, List
from PIL import Image

warnings.filterwarnings("ignore")

def ocr_preprocessor(image: Image)->Union[List[str], List[List[int]]]:
    """
    The goal of this function is to
    extract the bounding boxes and words
    from the given Image thanks to the
    Tesseract model
    
    Arguments:
        -image: Image: The input image
        to be preprocessed
    Retuns: 
        -words: List[str]: The words extracted
        from the input image
        -boxes: List[int]: The boxes containing
        the coordinates of the extracted words
    """
    
    ocr_data = pytesseract.image_to_boxes(image)

    words = []
    boxes = []

    for line in ocr_data.splitlines():
        box_data = line.split(' ')
        word = box_data[0]
        x, y, w, h = list(map(int, box_data[1:5]))
        
        words.append(word)
        boxes.append([x, y, w, h])
    
    return words, boxes

def prepare_layoutlm_input(words: List[str], boxes: List[List[int]], tokenizer, image_size: tuple):
    """
    Prepare data for LayoutLM
    
    Arguments:
    - words: List[str]: The words extracted from
    the Tesseract OCR process
    - boxes: List[List[int]]: The coordinates of
    extracted words
    - tokenizer: Le tokenizer LayoutLM de Hugging Face.
    - image_size: int: Image size
    
    Returns:
    - input_ids: The tokens of encoded words
    - attention_mask: Attention masks
    - bbox: Normalized coordinates of bounding
    boxes 
    """

    width, height = image_size

    encoding = tokenizer(words, is_split_into_words=True, padding="max_length", truncation=True, return_tensors="pt")

    normalized_boxes = []
    for word_idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        normalized_box = [
            int((x_min / width) * 1000), 
            int((y_min / height) * 1000),
            int((x_max / width) * 1000),
            int((y_max / height) * 1000),
        ]
        
        token_word_ids = encoding.word_ids()
        for token_word_id in token_word_ids:
            if token_word_id == word_idx:
                normalized_boxes.append(normalized_box)

    padded_boxes = normalized_boxes + [[0, 0, 0, 0]] * (encoding['input_ids'].size(1) - len(normalized_boxes))

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": torch.tensor(padded_boxes).unsqueeze(0)  # Ajouter une dimension pour le batch
    }

def visualize_ocr(image: Image, words: List[str], boxes: List[List[int]]) -> None:
    """
    Visualise the image with OCR results (bounding boxes and words)
    
    Arguments:
        - image: PIL.Image: The input image to display
        - words: List[str]: The words detected by the OCR
        - boxes: List[List[int]]: The bounding boxes for each word
    """
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for word, (x_min, y_min, x_max, y_max) in zip(words, boxes):
        rect_width = x_max - x_min
        rect_height = y_max - y_min
        
        rect = patches.Rectangle((x_min, image.size[1] - y_max), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        plt.text(x_min, image.size[1] - y_max - 5, word, color='blue', fontsize=8)
    
    plt.axis('off')
    plt.show()