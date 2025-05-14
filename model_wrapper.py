"""
Model wrapper module for SA-Test-Fix.

This module provides wrapper classes for different sentiment analysis models
to provide a unified interface for prediction and training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from typing import Dict, List, Tuple, Union, Optional

class ModelWrapper:
    """Base wrapper class for sentiment analysis models."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the model or model name
            device: Device to use for inference (cpu or cuda)
        """
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode by default
    
    def _load_model(self):
        """
        Load the model.
        
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, inputs):
        """
        Make predictions with the model.
        
        Args:
            inputs: Input data
            
        Returns:
            Model predictions
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def train_mode(self):
        """Set the model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def get_embeddings(self, inputs, apply_dropout=False):
        """
        Get embeddings from the model.
        
        Args:
            inputs: Input data
            apply_dropout: Whether to apply dropout
            
        Returns:
            Model embeddings
        """
        raise NotImplementedError("Subclasses must implement this method")


class BertModelWrapper(ModelWrapper):
    """Wrapper class for BERT-based sentiment analysis models."""
    
    def _load_model(self):
        """
        Load a BERT model.
        
        Returns:
            Loaded BERT model
        """
        try:
            # Try to load the model directly
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        except:
            # If that fails, try to load with a config file
            try:
                config = AutoConfig.from_pretrained(self.model_path)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_path, config=config)
            except:
                # If that also fails, create a new config
                config = AutoConfig.from_pretrained("bert-base-uncased")
                config.num_labels = 2  # Default to binary classification
                model = AutoModelForSequenceClassification.from_pretrained(self.model_path, config=config)
        
        return model
    
    def predict(self, inputs):
        """
        Make predictions with the BERT model.
        
        Args:
            inputs: Input data (can be tokenized dict or raw text)
            
        Returns:
            Model predictions (logits)
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, dict):
                # Handle tokenized inputs
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                outputs = self.model(**inputs)
            else:
                # Handle raw text inputs (should be tokenized first)
                raise ValueError("Inputs should be tokenized before prediction")
        
        return outputs.logits
    
    def get_embeddings(self, inputs, apply_dropout=False):
        """
        Get embeddings from the BERT model.
        
        Args:
            inputs: Input data
            apply_dropout: Whether to apply dropout
            
        Returns:
            Model embeddings
        """
        # Set dropout mode based on parameter
        if apply_dropout:
            self.model.train()
        else:
            self.model.eval()
        
        # Get embeddings
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden state (embeddings)
            embeddings = outputs.hidden_states[-1][:, 0, :]  # Use [CLS] token embedding
        else:
            raise ValueError("Inputs should be tokenized before getting embeddings")
        
        return embeddings


def load_model(model_path: str, model_type: str = "bert", device: str = None) -> ModelWrapper:
    """
    Load a model with the appropriate wrapper.
    
    Args:
        model_path: Path to the model or model name
        model_type: Type of model to load (bert, roberta, etc.)
        device: Device to use for inference (cpu or cuda)
        
    Returns:
        Wrapped model
    """
    if model_type.lower() in ["bert", "roberta", "distilbert"]:
        return BertModelWrapper(model_path, device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
