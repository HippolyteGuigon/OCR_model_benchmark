from datasets import load_dataset, DatasetDict

def load_data() -> DatasetDict:
    """
    The goal of this function
    is to load the funsd dataset
    
    Arguments:
        -None
    Returns:
        -dataset: DatasetDict: The 
        dataset once loaded
    """

    dataset = load_dataset("nielsr/funsd", trust_remote_code=True)

    return dataset
