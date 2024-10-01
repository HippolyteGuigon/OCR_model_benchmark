import pytesseract
import warnings

from typing import Union, List
from PIL import Image

from ocr_benchmark.utils.data_loading import load_data

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