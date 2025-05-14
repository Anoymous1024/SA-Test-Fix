"""
Genetic algorithm utilities for SA-Test-Fix.

This module provides utility functions for the genetic algorithm search framework,
including initialization, fitness evaluation, selection, crossover, and mutation.
"""

import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from tqdm import tqdm

class Individual:
    """Class representing an individual in the genetic algorithm."""

    def __init__(self, text: str, mutated_text: str = None, fitness: Tuple[float, float] = None,
                 mutator_name: str = None, relation_type: str = None):
        """
        Initialize an individual.

        Args:
            text: Original text
            mutated_text: Mutated text
            fitness: Fitness values (f1, f2)
            mutator_name: Name of the mutator used
            relation_type: Type of metamorphic relation
        """
        self.text = text
        self.mutated_text = mutated_text if mutated_text else text
        self.fitness = fitness if fitness else (0.0, 0.0)
        self.mutator_name = mutator_name
        self.relation_type = relation_type

    def __str__(self) -> str:
        """String representation of the individual."""
        return f"Original: {self.text}\nMutated: {self.mutated_text}\nFitness: {self.fitness}\nMutator: {self.mutator_name}\nRelation: {self.relation_type}"


def initialize_population(texts: List[str], population_size: int) -> List[Individual]:
    """
    Initialize a population of individuals.

    Args:
        texts: List of original texts
        population_size: Size of the population

    Returns:
        List of individuals
    """
    population = []

    # Ensure at least one individual per text
    for text in texts:
        population.append(Individual(text))

    # Fill the rest of the population with random texts
    while len(population) < population_size:
        text = random.choice(texts)
        population.append(Individual(text))

    return population


def evaluate_fitness(population: List[Individual], model, tokenizer,
                     semantic_similarity_fn: Callable) -> List[Individual]:
    """
    Evaluate the fitness of individuals in the population.

    Args:
        population: List of individuals
        model: Model for prediction
        tokenizer: Tokenizer for text processing
        semantic_similarity_fn: Function to calculate semantic similarity

    Returns:
        List of individuals with updated fitness values
    """
    for ind in tqdm(population, desc="Evaluating fitness"):
        if ind.mutated_text:
            # Tokenize texts
            original_encoded = tokenizer(
                ind.text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            mutated_encoded = tokenizer(
                ind.mutated_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # Get model predictions
            # Convert to the format expected by the model
            original_inputs = {k: v for k, v in original_encoded.items()}
            mutated_inputs = {k: v for k, v in mutated_encoded.items()}

            original_outputs = model.predict(original_inputs)
            mutated_outputs = model.predict(mutated_inputs)

            # Calculate f1: prediction probability difference
            original_probs = original_outputs.softmax(dim=1)
            mutated_probs = mutated_outputs.softmax(dim=1)
            f1 = float(torch.abs(original_probs - mutated_probs).max().item())

            # Calculate f2: semantic similarity
            f2 = semantic_similarity_fn(ind.text, ind.mutated_text)

            # Update fitness
            ind.fitness = (f1, f2)

    return population


def non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Perform non-dominated sorting of the population.

    Args:
        population: List of individuals

    Returns:
        List of fronts, where each front is a list of individuals
    """
    fronts = []
    dominated_count = {}
    dominated_solutions = {}

    for ind in population:
        dominated_count[ind] = 0
        dominated_solutions[ind] = []

    # Calculate domination
    for p in population:
        for q in population:
            if p == q:
                continue

            # Check if p dominates q
            if (p.fitness[0] > q.fitness[0] and p.fitness[1] >= q.fitness[1]) or \
               (p.fitness[0] >= q.fitness[0] and p.fitness[1] > q.fitness[1]):
                dominated_solutions[p].append(q)
            # Check if q dominates p
            elif (q.fitness[0] > p.fitness[0] and q.fitness[1] >= p.fitness[1]) or \
                 (q.fitness[0] >= p.fitness[0] and q.fitness[1] > p.fitness[1]):
                dominated_count[p] += 1

    # Create fronts
    current_front = []
    for ind in population:
        if dominated_count[ind] == 0:
            current_front.append(ind)

    fronts.append(current_front)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)

        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def calculate_crowding_distance(front: List[Individual]) -> Dict[Individual, float]:
    """
    Calculate crowding distance for individuals in a front.

    Args:
        front: List of individuals in a front

    Returns:
        Dictionary mapping individuals to their crowding distances
    """
    if len(front) <= 2:
        return {ind: float('inf') for ind in front}

    distances = {ind: 0.0 for ind in front}

    # Sort by each objective
    for i in range(2):  # Two objectives: f1 and f2
        sorted_front = sorted(front, key=lambda ind: ind.fitness[i])

        # Set boundary points to infinity
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        # Calculate distances for intermediate points
        f_min = sorted_front[0].fitness[i]
        f_max = sorted_front[-1].fitness[i]

        if f_max == f_min:
            continue

        for j in range(1, len(sorted_front) - 1):
            distances[sorted_front[j]] += (sorted_front[j+1].fitness[i] - sorted_front[j-1].fitness[i]) / (f_max - f_min)

    return distances


def select_parents(population: List[Individual], num_parents: int) -> List[Individual]:
    """
    Select parents for reproduction using tournament selection.

    Args:
        population: List of individuals
        num_parents: Number of parents to select

    Returns:
        List of selected parents
    """
    parents = []

    # Sort population into fronts
    fronts = non_dominated_sort(population)

    # Calculate crowding distances for each front
    crowding_distances = {}
    for front in fronts:
        front_distances = calculate_crowding_distance(front)
        crowding_distances.update(front_distances)

    # Tournament selection
    for _ in range(num_parents):
        # Select two random individuals
        candidates = random.sample(population, 2)

        # Find the fronts of the candidates
        candidate_fronts = []
        for candidate in candidates:
            for i, front in enumerate(fronts):
                if candidate in front:
                    candidate_fronts.append(i)
                    break

        # Select the better candidate
        if candidate_fronts[0] < candidate_fronts[1]:
            parents.append(candidates[0])
        elif candidate_fronts[0] > candidate_fronts[1]:
            parents.append(candidates[1])
        else:
            # If in the same front, select the one with larger crowding distance
            if crowding_distances[candidates[0]] > crowding_distances[candidates[1]]:
                parents.append(candidates[0])
            else:
                parents.append(candidates[1])

    return parents


def crossover(parents: List[Individual], crossover_rate: float = 0.8) -> List[Individual]:
    """
    Perform crossover between parents.

    Args:
        parents: List of parent individuals
        crossover_rate: Probability of crossover

    Returns:
        List of offspring individuals
    """
    offspring = []

    # Ensure even number of parents
    if len(parents) % 2 != 0:
        parents = parents[:-1]

    # Pair parents
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]

        if random.random() < crossover_rate:
            # Simple crossover: swap mutated texts
            child1 = Individual(parent1.text, parent2.mutated_text, mutator_name=parent2.mutator_name, relation_type=parent2.relation_type)
            child2 = Individual(parent2.text, parent1.mutated_text, mutator_name=parent1.mutator_name, relation_type=parent1.relation_type)

            offspring.extend([child1, child2])
        else:
            # No crossover, just copy parents
            offspring.extend([parent1, parent2])

    return offspring


def mutate(offspring: List[Individual], mutators: List, mutation_rate: float = 0.2) -> List[Individual]:
    """
    Mutate offspring individuals.

    Args:
        offspring: List of offspring individuals
        mutators: List of mutator objects
        mutation_rate: Probability of mutation

    Returns:
        List of mutated offspring individuals
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            # Select a random mutator
            mutator = random.choice(mutators)

            # Apply mutation
            mutated_texts = mutator.mutate(offspring[i].text)

            if mutated_texts:
                # Select a random mutated text
                mutated_text = random.choice(mutated_texts)

                # Update the individual
                offspring[i].mutated_text = mutated_text
                offspring[i].mutator_name = mutator.__class__.__name__
                offspring[i].relation_type = mutator.get_relation_type()

    return offspring


def select_next_generation(current_population: List[Individual], offspring: List[Individual],
                           population_size: int) -> List[Individual]:
    """
    Select individuals for the next generation.

    Args:
        current_population: Current population
        offspring: Offspring individuals
        population_size: Size of the population

    Returns:
        List of individuals for the next generation
    """
    # Combine current population and offspring
    combined_population = current_population + offspring

    # Sort into fronts
    fronts = non_dominated_sort(combined_population)

    # Calculate crowding distances for each front
    crowding_distances = {}
    for front in fronts:
        front_distances = calculate_crowding_distance(front)
        crowding_distances.update(front_distances)

    # Select individuals for the next generation
    next_generation = []
    i = 0

    while len(next_generation) + len(fronts[i]) <= population_size:
        next_generation.extend(fronts[i])
        i += 1

    # If we need more individuals, select from the next front based on crowding distance
    if len(next_generation) < population_size:
        remaining = population_size - len(next_generation)

        # Sort the next front by crowding distance
        sorted_front = sorted(fronts[i], key=lambda ind: crowding_distances[ind], reverse=True)

        # Add the required number of individuals
        next_generation.extend(sorted_front[:remaining])

    return next_generation
