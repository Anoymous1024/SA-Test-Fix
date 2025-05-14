"""
Main module for SA-Test-Fix.

This module provides the main entry point for the SA-Test-Fix tool.
"""

import os
import argparse
import torch
import configparser
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from tqdm import tqdm

from .data_handler import DataHandler
from .model_wrapper import load_model
from .metamorphic_mutators import (
    SynonymReplacementMutator,
    TenseChangeMutator,
    IntensifierMutator,
    ReducerMutator,
    PositivePhraseMutator,
    NegativePhraseMutator
)
from .ga_search import GASearch
from .detect import DefectDetector
from .sort_score import sort_samples
from .train import MultiStageTrainer
from .evaluator import Evaluator
from .utils import load_sentiment_dictionary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sa_test_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config

def get_mutators(sentiment_dict_path: str = None) -> List:
    """
    Get a list of mutator objects.
    
    Args:
        sentiment_dict_path: Path to the sentiment dictionary file
        
    Returns:
        List of mutator objects
    """
    # Load sentiment dictionary if provided
    sentiment_dict = None
    if sentiment_dict_path and os.path.exists(sentiment_dict_path):
        sentiment_dict = load_sentiment_dictionary(sentiment_dict_path)
    
    # Create mutators
    mutators = [
        SynonymReplacementMutator(sentiment_dict),
        TenseChangeMutator(sentiment_dict),
        IntensifierMutator(sentiment_dict),
        ReducerMutator(sentiment_dict),
        PositivePhraseMutator(sentiment_dict),
        NegativePhraseMutator(sentiment_dict)
    ]
    
    return mutators

def generate_test_cases(model, tokenizer, dataset, output_path: str, config: configparser.ConfigParser):
    """
    Generate test cases for a model.
    
    Args:
        model: Model to test
        tokenizer: Tokenizer for text processing
        dataset: Dataset to use for generating test cases
        output_path: Path to save the test cases
        config: Configuration object
    """
    # Get parameters from config
    population_size = config.getint('GA', 'population_size', fallback=50)
    num_generations = config.getint('GA', 'num_generations', fallback=20)
    crossover_rate = config.getfloat('GA', 'crossover_rate', fallback=0.8)
    mutation_rate = config.getfloat('GA', 'mutation_rate', fallback=0.2)
    
    # Get mutators
    sentiment_dict_path = config.get('Paths', 'sentiment_dict_path', fallback=None)
    mutators = get_mutators(sentiment_dict_path)
    
    # Create search framework
    search = GASearch(
        model=model,
        tokenizer=tokenizer,
        mutators=mutators,
        population_size=population_size,
        num_generations=num_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    
    # Generate test cases
    texts = [example['text'] for example in dataset]
    test_cases = search.generate_test_cases(texts)
    
    # Save test cases
    search.save_test_cases(test_cases, output_path)
    
    return test_cases

def detect_model_defects(model, test_cases_path: str, output_path: str = None):
    """
    Detect defects in a model using test cases.
    
    Args:
        model: Model to test
        test_cases_path: Path to the test cases file
        output_path: Path to save the defect cases
        
    Returns:
        Tuple of (defect_cases, defect_count)
    """
    # Load test cases
    data_handler = DataHandler()
    test_cases = data_handler.load_test_cases(test_cases_path)
    tokenized_test_cases = data_handler.tokenize_test_cases(test_cases)
    
    # Detect defects
    detector = DefectDetector(model)
    defect_cases, defect_count = detector.detect_defects(tokenized_test_cases)
    
    # Save defect cases if output path is provided
    if output_path and defect_cases:
        detector.save_defect_cases(defect_cases, output_path)
    
    # Print defect analysis
    detector.print_defect_analysis(defect_cases)
    
    return defect_cases, defect_count

def repair_model(model, defect_cases_path: str, output_path: str, config: configparser.ConfigParser):
    """
    Repair a model using defect cases.
    
    Args:
        model: Model to repair
        defect_cases_path: Path to the defect cases file
        output_path: Path to save the repaired model
        config: Configuration object
        
    Returns:
        Repaired model
    """
    # Get parameters from config
    total_epochs = config.getint('Training', 'total_epochs', fallback=30)
    rep_phase_ratio = config.getfloat('Training', 'rep_phase_ratio', fallback=0.3)
    joint_phase_ratio = config.getfloat('Training', 'joint_phase_ratio', fallback=0.4)
    initial_lr = config.getfloat('Training', 'initial_lr', fallback=2e-5)
    final_lr = config.getfloat('Training', 'final_lr', fallback=5e-6)
    temperature = config.getfloat('Training', 'temperature', fallback=0.1)
    initial_lambda = config.getfloat('Training', 'initial_lambda', fallback=0.1)
    final_lambda = config.getfloat('Training', 'final_lambda', fallback=0.9)
    rebuild_interval = config.getint('Training', 'rebuild_interval', fallback=1)
    batch_size = config.getint('Training', 'batch_size', fallback=16)
    
    # Load defect cases
    data_handler = DataHandler()
    defect_cases = data_handler.load_test_cases(defect_cases_path)
    tokenized_defect_cases = data_handler.tokenize_test_cases(defect_cases)
    
    # Split into train and validation sets
    train_ratio = 0.8
    train_size = int(len(tokenized_defect_cases) * train_ratio)
    train_cases = tokenized_defect_cases[:train_size]
    val_cases = tokenized_defect_cases[train_size:]
    
    # Sort train cases
    sorted_cases, sorted_scores = sort_samples(model, train_cases)
    
    # Create trainer
    trainer = MultiStageTrainer(
        model=model,
        total_epochs=total_epochs,
        rep_phase_ratio=rep_phase_ratio,
        joint_phase_ratio=joint_phase_ratio,
        initial_lr=initial_lr,
        final_lr=final_lr,
        temperature=temperature,
        initial_lambda=initial_lambda,
        final_lambda=final_lambda,
        rebuild_interval=rebuild_interval
    )
    
    # Train model
    repaired_model = trainer.train(
        train_samples=train_cases,
        val_samples=val_cases,
        sorted_samples=sorted_cases,
        sorted_scores=sorted_scores,
        batch_size=batch_size
    )
    
    # Save repaired model
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        repaired_model.model.save_pretrained(output_path)
        logger.info(f"Saved repaired model to {output_path}")
    
    return repaired_model

def evaluate_model(model, test_cases_path: str, original_model=None):
    """
    Evaluate a model using test cases.
    
    Args:
        model: Model to evaluate
        test_cases_path: Path to the test cases file
        original_model: Original model for comparison
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load test cases
    data_handler = DataHandler()
    test_cases = data_handler.load_test_cases(test_cases_path)
    tokenized_test_cases = data_handler.tokenize_test_cases(test_cases)
    
    # Create evaluator
    evaluator = Evaluator(model)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(tokenized_test_cases, original_model)
    
    # Print metrics
    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def main():
    """Main entry point for the SA-Test-Fix tool."""
    parser = argparse.ArgumentParser(description="SA-Test-Fix: Test and Fix Sentiment Analysis Models")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to the configuration file")
    parser.add_argument("--mode", type=str, choices=["test", "fix", "evaluate", "all"], default="all", help="Mode to run")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--test_cases_path", type=str, help="Path to the test cases file (for fix and evaluate modes)")
    parser.add_argument("--dataset_name", type=str, default="sst2", help="Name of the dataset to use (for test mode)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model_type = config.get('Model', 'model_type', fallback="bert")
    model = load_model(args.model_path, model_type)
    
    # Run in the specified mode
    if args.mode in ["test", "all"]:
        # Load dataset
        data_handler = DataHandler()
        dataset = data_handler.load_dataset(args.dataset_name, "test")
        
        # Generate test cases
        test_cases_path = os.path.join(args.output_dir, f"{args.dataset_name}_test_cases.txt")
        generate_test_cases(model, data_handler.tokenizer, dataset, test_cases_path, config)
        
        # Detect defects
        defect_cases_path = os.path.join(args.output_dir, f"{args.dataset_name}_defect_cases.txt")
        detect_model_defects(model, test_cases_path, defect_cases_path)
    
    if args.mode in ["fix", "all"]:
        # Use provided test cases path or the one from test mode
        test_cases_path = args.test_cases_path
        if not test_cases_path and args.mode == "all":
            test_cases_path = os.path.join(args.output_dir, f"{args.dataset_name}_defect_cases.txt")
        
        if not test_cases_path or not os.path.exists(test_cases_path):
            logger.error("Test cases path not provided or does not exist")
            return
        
        # Save original model for comparison
        original_model = model
        
        # Repair model
        repaired_model_path = os.path.join(args.output_dir, f"{args.dataset_name}_repaired_model")
        repaired_model = repair_model(model, test_cases_path, repaired_model_path, config)
    
    if args.mode in ["evaluate", "all"]:
        # Use provided test cases path or the one from test mode
        test_cases_path = args.test_cases_path
        if not test_cases_path and args.mode == "all":
            test_cases_path = os.path.join(args.output_dir, f"{args.dataset_name}_test_cases.txt")
        
        if not test_cases_path or not os.path.exists(test_cases_path):
            logger.error("Test cases path not provided or does not exist")
            return
        
        # Evaluate model
        if args.mode == "all":
            # Compare original and repaired models
            evaluate_model(repaired_model, test_cases_path, original_model)
        else:
            # Evaluate single model
            evaluate_model(model, test_cases_path)

if __name__ == "__main__":
    main()
