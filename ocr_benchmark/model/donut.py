import torch
import re

from transformers import DonutProcessor, VisionEncoderDecoderModel
from ocr_benchmark.preprocessing.preprocessing import image_preprocessing_donut


def predict_with_donut(image_path: str, prompt: str):
    """
    Faire une prédiction sur une image en utilisant Donut.
    """

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base"
    )

    pixel_values = image_preprocessing_donut(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_values = pixel_values.to(device)
    model.to(device)

    prompt = "<s_rvlcdip>"

    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    return processor.token2json(sequence)
