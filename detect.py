"""
Defect detection module for SA-Test-Fix.

This module implements defect detection for sentiment analysis models
using metamorphic testing.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

class DefectDetector:
    """Defect detector for sentiment analysis models."""
    
    def __init__(self, model, device=None):
        """
        Initialize the defect detector.
        
        Args:
            model: Model to test
            device: Device to use for testing
        """
        self.model = model
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    def detect_defects(self, test_cases: List[Tuple], verbose: bool = True) -> Tuple[List[Tuple], int]:
        """
        Detect defects in a model using test cases.
        
        Args:
            test_cases: List of test cases (original_text, mutated_text, relation_type, test_name)
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (defect_cases, defect_count)
        """
        self.model.eval_mode()
        defect_cases = []
        defect_count = 0
        
        if verbose:
            test_cases_iter = tqdm(test_cases, desc="Detecting defects")
        else:
            test_cases_iter = test_cases
        
        for x, x_prime, label, relation_type, test_name in test_cases_iter:
            # Get predictions
            x_logits = self.model.predict(x)
            x_prime_logits = self.model.predict(x_prime)
            
            x_pred = torch.argmax(x_logits, dim=1).item()
            x_prime_pred = torch.argmax(x_prime_logits, dim=1).item()
            
            # Check if the prediction violates the metamorphic relation
            is_defect = False
            
            if relation_type == "identity":
                # For identity relation, predictions should be the same
                if x_pred != x_prime_pred:
                    is_defect = True
            else:  # inequality relation
                # For inequality relation, check the direction of change based on test name
                if "Positive" in test_name and x_prime_pred < x_pred:
                    is_defect = True
                elif "Negative" in test_name and x_prime_pred > x_pred:
                    is_defect = True
                elif "Intensifier" in test_name and x_prime_pred < x_pred:
                    is_defect = True
                elif "Reducer" in test_name and x_prime_pred > x_pred:
                    is_defect = True
            
            if is_defect:
                defect_cases.append((x, x_prime, label, relation_type, test_name))
                defect_count += 1
        
        if verbose:
            print(f"Detected {defect_count} defects out of {len(test_cases)} test cases")
        
        return defect_cases, defect_count
    
    def calculate_error_rate(self, test_cases: List[Tuple]) -> float:
        """
        Calculate the error rate of a model on test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Error rate
        """
        _, defect_count = self.detect_defects(test_cases, verbose=False)
        
        return defect_count / len(test_cases) if test_cases else 0.0
    
    def save_defect_cases(self, defect_cases: List[Tuple], file_path: str, include_label: bool = True):
        """
        Save defect cases to a file.
        
        Args:
            defect_cases: List of defect cases
            file_path: Path to save the defect cases
            include_label: Whether to include the label in the output
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for x, x_prime, label, relation_type, test_name in defect_cases:
                # Get original text and mutated text
                if isinstance(x, dict):
                    # Handle tokenized inputs
                    original_text = self.model.tokenizer.decode(x["input_ids"], skip_special_tokens=True)
                    mutated_text = self.model.tokenizer.decode(x_prime["input_ids"], skip_special_tokens=True)
                else:
                    # Handle raw text inputs
                    original_text = x
                    mutated_text = x_prime
                
                f.write(f"Original: {original_text}\n")
                f.write(f"Perturbed: {mutated_text}\n")
                
                if include_label and label is not None:
                    f.write(f"Label: {label}\n")
                
                f.write(f"Type: {'INV' if relation_type == 'identity' else 'DIR'}\n")
                f.write(f"Test: {test_name}\n")
                f.write("\n")
        
        print(f"Saved {len(defect_cases)} defect cases to {file_path}")
    
    def analyze_defects(self, defect_cases: List[Tuple]) -> Dict[str, Dict[str, int]]:
        """
        Analyze defect cases by test type and relation type.
        
        Args:
            defect_cases: List of defect cases
            
        Returns:
            Dictionary of defect statistics
        """
        stats = {
            "by_test_type": {},
            "by_relation_type": {
                "identity": 0,
                "inequality": 0
            }
        }
        
        for _, _, _, relation_type, test_name in defect_cases:
            # Count by relation type
            if relation_type == "identity":
                stats["by_relation_type"]["identity"] += 1
            else:
                stats["by_relation_type"]["inequality"] += 1
            
            # Count by test type
            if test_name not in stats["by_test_type"]:
                stats["by_test_type"][test_name] = 0
            
            stats["by_test_type"][test_name] += 1
        
        return stats
    
    def print_defect_analysis(self, defect_cases: List[Tuple]):
        """
        Print analysis of defect cases.
        
        Args:
            defect_cases: List of defect cases
        """
        stats = self.analyze_defects(defect_cases)
        
        print("Defect Analysis:")
        print("----------------")
        print(f"Total defects: {len(defect_cases)}")
        
        print("\nBy Relation Type:")
        for relation_type, count in stats["by_relation_type"].items():
            print(f"  {relation_type}: {count}")
        
        print("\nBy Test Type:")
        for test_type, count in sorted(stats["by_test_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {test_type}: {count}")


def detect_defects(model, test_cases: List[Tuple], device=None, verbose: bool = True) -> Tuple[List[Tuple], int]:
    """
    Detect defects in a model using test cases.
    
    Args:
        model: Model to test
        test_cases: List of test cases
        device: Device to use for testing
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (defect_cases, defect_count)
    """
    detector = DefectDetector(model, device)
    return detector.detect_defects(test_cases, verbose)
