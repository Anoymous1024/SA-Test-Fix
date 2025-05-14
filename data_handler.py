"""
Data handler module for SA-Test-Fix.

This module is responsible for loading and preprocessing datasets for sentiment analysis
testing and fixing.
"""

import os
import torch
from typing import Dict, List, Tuple, Union, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
import spacy
import numpy as np
from tqdm import tqdm

# Load spaCy model for text processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class DataHandler:
    """Data handler for loading and preprocessing datasets."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", max_length: int = 128):
        """
        Initialize the data handler.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def load_dataset(self, dataset_name: str, split: str = "train") -> Dict:
        """
        Load a dataset from Hugging Face datasets.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Split of the dataset to load (train, validation, test)
            
        Returns:
            Dictionary containing the dataset
        """
        if dataset_name == "sst2":
            dataset = load_dataset("SetFit/sst2", split=split)
        elif dataset_name == "sst5":
            dataset = load_dataset("SetFit/sst5", split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        return dataset
    
    def tokenize_dataset(self, dataset, text_column: str = "text", label_column: str = "label") -> Dict:
        """
        Tokenize a dataset.
        
        Args:
            dataset: Dataset to tokenize
            text_column: Name of the text column
            label_column: Name of the label column
            
        Returns:
            Dictionary containing the tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(
            [col for col in tokenized_dataset.column_names if col != label_column and col not in ["input_ids", "attention_mask", "token_type_ids"]]
        )
        
        return tokenized_dataset
    
    def load_test_cases(self, file_path: str) -> List[Tuple]:
        """
        Load test cases from a file.
        
        Args:
            file_path: Path to the test cases file
            
        Returns:
            List of tuples containing (original_text, perturbed_text, label, relation_type, test_name)
        """
        test_cases = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Original:"):
                original_text = line[len("Original:"):].strip()
                
                # Get perturbed text
                i += 1
                if i < len(lines) and lines[i].strip().startswith("Perturbed:"):
                    perturbed_text = lines[i].strip()[len("Perturbed:"):].strip()
                else:
                    i += 1
                    continue
                
                # Get label if available
                label = None
                if i+1 < len(lines) and "Label:" in lines[i+1]:
                    label_part = lines[i+1].strip().split("Label:")
                    if len(label_part) > 1:
                        try:
                            label = int(label_part[1].strip())
                            i += 1
                        except ValueError:
                            pass
                
                # Get relation type and test name if available
                relation_type = "identity"  # Default relation type
                test_name = "Unknown Test"
                
                if i+1 < len(lines) and "Type:" in lines[i+1]:
                    type_parts = lines[i+1].strip().split("Type:")
                    if len(type_parts) > 1:
                        type_value = type_parts[1].strip()
                        relation_type = "identity" if type_value == "INV" else "inequality"
                        i += 1
                
                if i+1 < len(lines) and "Test:" in lines[i+1]:
                    test_parts = lines[i+1].strip().split("Test:")
                    if len(test_parts) > 1:
                        test_name = test_parts[1].strip()
                        i += 1
                
                test_cases.append((original_text, perturbed_text, label, relation_type, test_name))
            
            i += 1
        
        return test_cases
    
    def tokenize_test_cases(self, test_cases: List[Tuple]) -> List[Tuple]:
        """
        Tokenize test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            List of tokenized test cases
        """
        tokenized_test_cases = []
        
        for original_text, perturbed_text, label, relation_type, test_name in tqdm(test_cases, desc="Tokenizing test cases"):
            original_encoded = self.tokenizer(
                original_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            perturbed_encoded = self.tokenizer(
                perturbed_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Convert to dict with single tensors
            original_encoded = {k: v.squeeze(0) for k, v in original_encoded.items()}
            perturbed_encoded = {k: v.squeeze(0) for k, v in perturbed_encoded.items()}
            
            tokenized_test_cases.append((
                original_encoded,
                perturbed_encoded,
                label,
                relation_type,
                test_name
            ))
        
        return tokenized_test_cases
