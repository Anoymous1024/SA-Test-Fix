"""
Genetic algorithm search framework for SA-Test-Fix.

This module implements a genetic algorithm-based search framework for generating
high-quality test cases for sentiment analysis models.
"""

import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from tqdm import tqdm

from .ga_utils import (
    Individual, 
    initialize_population, 
    evaluate_fitness, 
    select_parents, 
    crossover, 
    mutate, 
    select_next_generation
)
from .metamorphic_mutators import MetamorphicMutator
from .utils import calculate_semantic_similarity

class GASearch:
    """Genetic algorithm search framework for generating test cases."""
    
    def __init__(self, model, tokenizer, mutators: List[MetamorphicMutator], 
                 population_size: int = 50, num_generations: int = 20,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 semantic_similarity_fn: Callable = None):
        """
        Initialize the search framework.
        
        Args:
            model: Model for prediction
            tokenizer: Tokenizer for text processing
            mutators: List of mutator objects
            population_size: Size of the population
            num_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            semantic_similarity_fn: Function to calculate semantic similarity
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mutators = mutators
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.semantic_similarity_fn = semantic_similarity_fn or calculate_semantic_similarity
    
    def search(self, texts: List[str], verbose: bool = True) -> List[Tuple[str, str, str, str]]:
        """
        Search for high-quality test cases.
        
        Args:
            texts: List of original texts
            verbose: Whether to print progress information
            
        Returns:
            List of tuples containing (original_text, mutated_text, mutator_name, relation_type)
        """
        # Initialize population
        population = initialize_population(texts, self.population_size)
        
        # Apply initial mutations
        for ind in population:
            # Select a random mutator
            mutator = random.choice(self.mutators)
            
            # Apply mutation
            mutated_texts = mutator.mutate(ind.text)
            
            if mutated_texts:
                # Select a random mutated text
                mutated_text = random.choice(mutated_texts)
                
                # Update the individual
                ind.mutated_text = mutated_text
                ind.mutator_name = mutator.__class__.__name__
                ind.relation_type = mutator.get_relation_type()
        
        # Evaluate initial population
        population = evaluate_fitness(population, self.model, self.tokenizer, self.semantic_similarity_fn)
        
        # Main loop
        for generation in range(self.num_generations):
            if verbose:
                print(f"Generation {generation+1}/{self.num_generations}")
            
            # Select parents
            parents = select_parents(population, self.population_size)
            
            # Perform crossover
            offspring = crossover(parents, self.crossover_rate)
            
            # Perform mutation
            offspring = mutate(offspring, self.mutators, self.mutation_rate)
            
            # Evaluate offspring
            offspring = evaluate_fitness(offspring, self.model, self.tokenizer, self.semantic_similarity_fn)
            
            # Select next generation
            population = select_next_generation(population, offspring, self.population_size)
            
            if verbose:
                # Print statistics
                avg_f1 = np.mean([ind.fitness[0] for ind in population])
                avg_f2 = np.mean([ind.fitness[1] for ind in population])
                print(f"  Average f1: {avg_f1:.4f}, Average f2: {avg_f2:.4f}")
        
        # Return the final population as test cases
        test_cases = []
        for ind in population:
            if ind.mutated_text and ind.mutated_text != ind.text:
                test_cases.append((ind.text, ind.mutated_text, ind.mutator_name, ind.relation_type))
        
        return test_cases
    
    def generate_test_cases(self, texts: List[str], verbose: bool = True) -> List[Dict]:
        """
        Generate test cases in a structured format.
        
        Args:
            texts: List of original texts
            verbose: Whether to print progress information
            
        Returns:
            List of dictionaries containing test case information
        """
        # Search for test cases
        raw_test_cases = self.search(texts, verbose)
        
        # Convert to structured format
        test_cases = []
        for original_text, mutated_text, mutator_name, relation_type in raw_test_cases:
            test_case = {
                "original_text": original_text,
                "mutated_text": mutated_text,
                "mutator": mutator_name,
                "relation_type": relation_type,
                "test_name": f"{mutator_name} Test"
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def save_test_cases(self, test_cases: List[Dict], file_path: str):
        """
        Save test cases to a file.
        
        Args:
            test_cases: List of test case dictionaries
            file_path: Path to save the test cases
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for tc in test_cases:
                f.write(f"Original: {tc['original_text']}\n")
                f.write(f"Perturbed: {tc['mutated_text']}\n")
                f.write(f"Type: {'INV' if tc['relation_type'] == 'identity' else 'DIR'}\n")
                f.write(f"Test: {tc['test_name']}\n")
                f.write("\n")
        
        print(f"Saved {len(test_cases)} test cases to {file_path}")
    
    def load_test_cases(self, file_path: str) -> List[Dict]:
        """
        Load test cases from a file.
        
        Args:
            file_path: Path to the test cases file
            
        Returns:
            List of test case dictionaries
        """
        test_cases = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("Original:"):
                original_text = line[len("Original:"):].strip()
                
                # Get mutated text
                i += 1
                if i < len(lines) and lines[i].strip().startswith("Perturbed:"):
                    mutated_text = lines[i].strip()[len("Perturbed:"):].strip()
                else:
                    i += 1
                    continue
                
                # Get relation type
                relation_type = "identity"  # Default relation type
                i += 1
                if i < len(lines) and lines[i].strip().startswith("Type:"):
                    type_value = lines[i].strip()[len("Type:"):].strip()
                    relation_type = "identity" if type_value == "INV" else "inequality"
                else:
                    i += 1
                    continue
                
                # Get test name
                test_name = "Unknown Test"
                i += 1
                if i < len(lines) and lines[i].strip().startswith("Test:"):
                    test_name = lines[i].strip()[len("Test:"):].strip()
                
                # Create test case dictionary
                test_case = {
                    "original_text": original_text,
                    "mutated_text": mutated_text,
                    "relation_type": relation_type,
                    "test_name": test_name,
                    "mutator": test_name.replace(" Test", "")
                }
                
                test_cases.append(test_case)
            
            i += 1
        
        return test_cases
