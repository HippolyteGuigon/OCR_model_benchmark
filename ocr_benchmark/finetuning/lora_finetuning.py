from ocr_benchmark.preprocessing.preprocessing import preprocess_funsd

from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMv2Processor,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType

from datasets import load_dataset


def lora_fintenuning(r: int = 16, epochs: int = 3, **kwargs):
    """
    The goal of this function is to fine-tune a LayoutLM model using LoRA
    (Low-Rank Adaptation) for token classification. The function configures
    LoRA, loads the dataset, and performs the fine-tuning process using Hugging
    Face's Trainer class.

    Arguments:
        - r: int: The low-rank dimension used in the LoRA
        configuration. The default value is 16.
        - epochs: int: The number of training epochs.
        Default is set to 3.
        - kwargs: Additional parameters such as
        a pre-initialized model or processor
        that can be passed to override the defaults.
        If not provided, defaults are used for both
        the model and processor.

    Returns:
        None
    """

    if "model" not in kwargs.keys():
        model = LayoutLMForTokenClassification.from_pretrained(
            "microsoft/layoutlm-base-uncased", num_labels=7
        )
    else:
        model = kwargs["model"]

    if "processor" not in kwargs.keys():
        processor = LayoutLMv2Processor.from_pretrained(
            "microsoft/layoutlmv2-base-uncased", apply_ocr=False
        )
    else:
        processor = kwargs["processor"]

    target_modules = ["qkv_linear", "output.dense"]
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    dataset = load_dataset("nielsr/funsd", split="train")
    dataset = dataset.map(preprocess_funsd, batched=False)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
    )

    trainer.train()
