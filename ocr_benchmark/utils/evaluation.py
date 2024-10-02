import pandas as pd
from sklearn.metrics import classification_report


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
