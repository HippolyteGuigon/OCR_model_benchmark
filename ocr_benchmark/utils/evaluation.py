from sklearn.metrics import classification_report


def evaluate(predictions, true_labels):
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

    pred_flat = [pred for sublist in predictions for pred in sublist]
    labels_flat = [label for sublist in true_labels for label in sublist]

    report = classification_report(
        labels_flat, pred_flat, target_names=label_list
    )

    return report
