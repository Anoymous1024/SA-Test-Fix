"""
Utility functions for SA-Test-Fix.

This module provides utility functions for text processing, similarity calculation,
and syntactic analysis.
"""

import os
import torch
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm

# Load spaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get stopwords
STOPWORDS = set(stopwords.words('english'))

def calculate_so_value(text: str, sentiment_dict: Dict[str, float]) -> float:
    """
    Calculate the Semantic Orientation (SO) value of a text.
    
    Args:
        text: Input text
        sentiment_dict: Dictionary mapping words to sentiment values
        
    Returns:
        SO value of the text
    """
    # Parse the text with spaCy
    doc = nlp(text)
    
    # Calculate SO value
    so_value = 0.0
    for token in doc:
        word = token.text.lower()
        if word in sentiment_dict:
            so_value += sentiment_dict[word]
    
    return so_value

def calculate_tree_edit_distance(doc1, doc2) -> int:
    """
    Calculate the tree edit distance between two dependency trees.
    
    Args:
        doc1: First spaCy doc
        doc2: Second spaCy doc
        
    Returns:
        Tree edit distance
    """
    # Simplified implementation - count differences in dependency relations
    distance = 0
    
    # Get dependency trees
    tree1 = [(token.text, token.dep_, token.head.text) for token in doc1]
    tree2 = [(token.text, token.dep_, token.head.text) for token in doc2]
    
    # Count differences
    distance = len(set(tree1) ^ set(tree2))
    
    return distance

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using spaCy.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    return doc1.similarity(doc2)

def filter_stopwords(text: str) -> str:
    """
    Filter stopwords from a text.
    
    Args:
        text: Input text
        
    Returns:
        Text with stopwords removed
    """
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if token.text.lower() not in STOPWORDS]
    
    return " ".join(filtered_tokens)

def get_top_k_sentiment_words(text: str, sentiment_dict: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top k words with the highest absolute sentiment values.
    
    Args:
        text: Input text
        sentiment_dict: Dictionary mapping words to sentiment values
        k: Number of top words to return
        
    Returns:
        List of (word, sentiment_value) tuples
    """
    doc = nlp(text)
    
    # Get sentiment values for words in the text
    word_sentiments = []
    for token in doc:
        word = token.text.lower()
        if word in sentiment_dict and word not in STOPWORDS:
            word_sentiments.append((word, sentiment_dict[word]))
    
    # Sort by absolute sentiment value
    word_sentiments.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return word_sentiments[:k]

def load_sentiment_dictionary(file_path: str) -> Dict[str, float]:
    """
    Load a sentiment dictionary from a file.
    
    Args:
        file_path: Path to the sentiment dictionary file
        
    Returns:
        Dictionary mapping words to sentiment values
    """
    sentiment_dict = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = parts[0].lower()
                try:
                    value = float(parts[1])
                    sentiment_dict[word] = value
                except ValueError:
                    continue
    
    return sentiment_dict

def apply_negation_operation(so_value: float, negation_strength: float = 0.5) -> float:
    """
    Apply negation operation to a SO value.
    
    Args:
        so_value: Original SO value
        negation_strength: Strength of the negation operation
        
    Returns:
        Modified SO value
    """
    if so_value > 0:
        return so_value - negation_strength
    else:
        return so_value + negation_strength

def apply_intensification_operation(so_value: float, intensification_factor: float = 1.5) -> float:
    """
    Apply intensification operation to a SO value.
    
    Args:
        so_value: Original SO value
        intensification_factor: Factor to intensify the SO value
        
    Returns:
        Modified SO value
    """
    return so_value * intensification_factor

def calculate_fluency_score(text: str) -> float:
    """
    Calculate a fluency score for a text.
    
    Args:
        text: Input text
        
    Returns:
        Fluency score between 0 and 1
    """
    doc = nlp(text)
    
    # Simple heuristic: ratio of valid dependency relations
    valid_deps = sum(1 for token in doc if token.dep_ != "")
    total_tokens = len(doc)
    
    if total_tokens == 0:
        return 0.0
    
    return valid_deps / total_tokens
