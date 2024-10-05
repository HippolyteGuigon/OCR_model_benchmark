import pandas as pd

from tqdm import tqdm
from sklearn.metrics import classification_report

from ocr_benchmark.utils.data_loading import load_data
from ocr_benchmark.preprocessing.preprocessing import image_preprocessing
from ocr_benchmark.model.layoutlm import predict


def evaluate(predictions, true_labels) -> pd.DataFrame:
    """
    Evaluate the predictions against the true labels using common NER metrics:
    precision, recall, and F1-score.

    Arguments:
        - predictions: List[List[int]]: The predicted labels for each token.
        - true_labels: List[List[int]]: The true labels for each token.

    Returns:
        - report: str: A classification report with precision,
        recall, and F1-score.
    """

    label_list = [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]

    id2label = {
        0: "O",
        1: "B-HEADER",
        2: "I-HEADER",
        3: "B-QUESTION",
        4: "I-QUESTION",
        5: "B-ANSWER",
        6: "I-ANSWER",
    }

    predictions = [id2label[i] for i in predictions]
    true_labels = [id2label[i] for i in true_labels]

    report_dict = classification_report(
        true_labels, predictions, target_names=label_list, output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    return report_df


def full_evaluation(**kwargs) -> pd.DataFrame:
    """
    The goal of this function is to
    evaluate the LayoutLM model on the
    entire FUNSD dataset

    Arguments:
        -None
    Returns:
        -report_df: pd.DataFrame:
        The full Dataframe containing
        evaluation
    """

    if "model" in kwargs.keys():
        model = kwargs["model"]

    dataset = load_data()
    test_set = dataset["test"]

    all_true_labels = []
    all_predictions = []

    for image_index in tqdm(range(len(test_set))):
        try:
            encoding, info = image_preprocessing(image_index=image_index)
            prediction = predict(encoding=encoding, model=model)

            _, _, _, true_labels = info
        except RuntimeError:
            continue

        all_predictions.extend(prediction.tolist())
        all_true_labels.extend(true_labels)

    label_list = [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]

    id2label = {
        0: "O",
        1: "B-HEADER",
        2: "I-HEADER",
        3: "B-QUESTION",
        4: "I-QUESTION",
        5: "B-ANSWER",
        6: "I-ANSWER",
    }

    all_predictions = [id2label[i] for i in all_predictions]
    all_true_labels = [id2label[i] for i in all_true_labels]

    report_dict = classification_report(
        all_true_labels,
        all_predictions,
        target_names=label_list,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    return report_df
