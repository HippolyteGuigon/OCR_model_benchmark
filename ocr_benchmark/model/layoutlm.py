import torch

from typing import Any, Dict
from transformers import LayoutLMv2ForTokenClassification


def predict(encoding: Dict[str, Any], **kwargs) -> torch.Tensor:
    """
    The purpose of this function is to make
    predictions using the LayoutLMv2 model
    based on the given input encoding.

    Arguments:
        - encoding: Dict[str, Any]: A dictionary containing
          the encoded input for the LayoutLMv2 model, which includes
          the image, tokens, bounding boxes, etc.
        - **kwargs: optional keyword arguments:
            - model: A preloaded LayoutLMv2ForTokenClassification
              model (if not provided, the function will load a
              pre-trained LayoutLMv2 model with 7 labels).

    Returns:
        - predictions: torch.Tensor: A tensor containing the
        predicted labels for each token in the input.

    Notes:
        - The model is executed in evaluation mode without gradient
        computation (`torch.no_grad()`), which ensures that the
        predictions are made efficiently without updating the
        model's parameters.
        - If no model is provided via `kwargs`, a pre-trained
        `LayoutLMv2ForTokenClassification` model is loaded.
    """

    if "model" not in kwargs.keys():
        model = LayoutLMv2ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", num_labels=7
        )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    predictions = predictions.squeeze(0)

    valid_predictions = predictions[encoding["labels"] != -100]

    return valid_predictions
