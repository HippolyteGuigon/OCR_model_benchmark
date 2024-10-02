import torch

from transformers import LayoutLMv2ForTokenClassification

def predict(encoding, **kwargs):

    if "model" not in kwargs.keys():
       model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=7)

    with torch.no_grad():
     outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    return predictions