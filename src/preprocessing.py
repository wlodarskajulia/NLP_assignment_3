from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np

def basic_text_cleaning(text: str) -> list[str]:
    """
    Minimal cleaning is applied and tokenization:
            - Lowercasing
            - Split on whitespace to obtain word tokens

    Args:
            text(str): Input string

    Returns:
            list(str): List of preprocessed word tokens.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return tokens


def create_vocabulary(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    """
    The function builds a word-to-index vocabulary from a list of texts. Special
    tokens <PAD> (0) and <UNK> (1) are always included. Only the words appearing at least
    min_freq times are added.

    Args:
            texts (list[str]): List of raw input strings
            min_freq (int): Minimum frequency for a word to be included.

    Returns:
            dict[str, int]: Mapping from word to integer index.
    """
    # Count tokens
    counter = Counter()
    for text in texts:
            tokens = basic_text_cleaning(text)
            counter.update(tokens)

    # Vocabulary
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    
    for word, freq in counter.items():
            if freq >= min_freq:
                    vocabulary[word] = len(vocabulary)

    return vocabulary


def text_to_sequence(tokens: list[str], vocabulary: dict[str, int]) -> list[int]:
    """
    Convert a list of tokens to a list of vocabulary indices.

    Args:
            tokens (list[str]): List of word tokens
            vocabulary (dict[str, int]): Word-to-index vocabulary mapping.
    
    Returns:
            list[int]: List of integers corresponding to input tokens.
    """
    return [vocabulary.get(token, 1) for token in tokens]


def max_len_calculations(sequences: list[list[int]], percentile: int = 90) -> int:
    """
    Computes a maximum length based on a given percentile.

    Args:
            sequences (list[list[int]]): List of tokenized and indexed sequences.
            percentile (int): Percentile that determines the cutoff length.
    
    Returns:
            int: The sequence length at the given percentile.
    """
    lengths = [len(seq) for seq in sequences]
    print("Average length:", sum(lengths)/len(lengths))
    print("Max length:", max(lengths))
    max_len = int(np.percentile(lengths, percentile))
    return max_len


def plot_sequence_length_distribution(sequences: list[list[int]], percentile: int = 90) -> None:
    """
    Plot a histogram of sequence lengths with markers for the mean and 
    the specified percentile cutoff.

    Args:
            sequences (list[list[int]]): List of tokenized and indexed sequences.
            percentile (int): Percentile to display as a cutoff marker. Defaults to 90.

    Returns:
            None
    """
    lengths = [len(seq) for seq in sequences]
    cutoff = int(np.percentile(lengths, percentile))
    mean = sum(lengths) / len(lengths)

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=40, color="steelblue", edgecolor="white")
    plt.axvline(mean, color="orange", linestyle="--", label=f"Mean ({mean:.1f})")
    plt.axvline(cutoff, color="red", linestyle="--", label=f"{percentile}th percentile ({cutoff})")
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.title("Distribution of sequence lengths (training set)")
    plt.legend()
    plt.tight_layout()
    plt.show()