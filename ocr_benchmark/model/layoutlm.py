import torch

from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from transformers import pipeline


def predict(layoutlm_inputs)->torch.Tensor:
    
    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=layoutlm_inputs["input_ids"],
            attention_mask=layoutlm_inputs["attention_mask"],
            bbox=layoutlm_inputs["bbox"]
        )

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    return predictions