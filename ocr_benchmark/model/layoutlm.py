import torch

from transformers import LayoutLMv2ForTokenClassification

id2label = {0: "O", 1: "B-HEADER", 2: "I-HEADER", 3: "B-QUESTION", 4: "I-QUESTION", 5: "B-ANSWER", 6: "I-ANSWER"}

def predict(encoding, **kwargs):

    if "model" not in kwargs.keys():
       model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=7)

    with torch.no_grad():
     outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    return predictions