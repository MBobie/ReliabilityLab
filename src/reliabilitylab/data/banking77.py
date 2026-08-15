"""Loader for the BANKING77 intent-classification dataset."""

from datasets import DatasetDict, load_dataset


def load_banking77() -> DatasetDict:
    """Load BANKING77 from the DeepPavlov Hugging Face mirror.

    The project uses the script-free Parquet mirror to remain compatible
    with the installed Hugging Face datasets stack.

    Returns
    -------
    DatasetDict
        Dataset containing "train" and "test" splits. Each example
        contains an "utterance" string and an integer "label".
    """
    return load_dataset("DeepPavlov/banking77")