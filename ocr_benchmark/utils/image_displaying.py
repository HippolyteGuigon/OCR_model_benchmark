import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from typing import List

def result_display(image_path: str, boxes: List, annotations, predictions):
    image = Image.open(image_path)

    max_size = (1000, 1000)  
    image.thumbnail(max_size, Image.ANTIALIAS)

    image_width, image_height = image.size

    id2label = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
    label2color = {'HEADER': 'blue', 'QUESTION': 'green', 'ANSWER': 'orange', 'O': 'black'}

    scaled_boxes = [(int(box[0] / 1000 * image_width), 
                    int(box[1] / 1000 * image_height), 
                    int(box[2] / 1000 * image_width), 
                    int(box[3] / 1000 * image_height)) for box in boxes]

    fig, ax = plt.subplots(1, figsize=(image_width / 100, image_height / 100))  # Ajuster la taille de la figure
    ax.imshow(image)

    for token, box, label_id in zip(annotations, scaled_boxes, predictions[0].tolist()):
        label = id2label[label_id]
        label_class = label.split('-')[-1] if '-' in label else label
        color = label2color.get(label_class, 'red')

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        plt.text(box[0], box[1] - 10, label, color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
