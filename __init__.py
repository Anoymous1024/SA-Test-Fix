"""
SA-Test-Fix: A tool for testing and fixing sentiment analysis models.

This package provides tools for generating test cases, detecting defects,
and repairing sentiment analysis models using metamorphic testing and
contrastive learning.
"""

from .data_handler import DataHandler
from .model_wrapper import load_model, ModelWrapper, BertModelWrapper
from .metamorphic_mutators import (
    MetamorphicMutator,
    SynonymReplacementMutator,
    TenseChangeMutator,
    IntensifierMutator,
    ReducerMutator,
    PositivePhraseMutator,
    NegativePhraseMutator
)
from .ga_search import GASearch
from .detect import DefectDetector, detect_defects
from .sort_score import sort_samples, select_top_k_samples
from .train import MultiStageTrainer
from .model_retrainer import ModelRetrainer, ContrastiveLoss, MixedLoss
from .evaluator import Evaluator
from .utils import (
    calculate_so_value,
    calculate_tree_edit_distance,
    calculate_semantic_similarity,
    filter_stopwords,
    get_top_k_sentiment_words,
    load_sentiment_dictionary,
    apply_negation_operation,
    apply_intensification_operation,
    calculate_fluency_score
)

__version__ = "0.1.0"
