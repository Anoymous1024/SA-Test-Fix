"""
Metamorphic mutators module for SA-Test-Fix.

This module implements various mutation operators for metamorphic testing
of sentiment analysis models, including both general text mutators and
sentiment-specific mutators.
"""

import random
import re
import spacy
import nltk
from nltk.corpus import wordnet
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
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
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Constants
INTENSIFIERS = ["very", "extremely", "really", "quite", "so", "absolutely", "incredibly", "highly", "totally"]
REDUCERS = ["somewhat", "slightly", "a bit", "a little", "rather", "kind of", "sort of", "barely"]
POSITIVE_PHRASES = [
    "I enjoyed it.", 
    "It was great.", 
    "This is excellent.", 
    "I liked it a lot.",
    "This is wonderful."
]
NEGATIVE_PHRASES = [
    "I didn't like it.", 
    "It was terrible.", 
    "This is awful.", 
    "I hated it.",
    "This is disappointing."
]

class MetamorphicMutator:
    """Base class for metamorphic mutators."""
    
    def __init__(self, sentiment_dict: Dict[str, float] = None):
        """
        Initialize the mutator.
        
        Args:
            sentiment_dict: Dictionary mapping words to sentiment values
        """
        self.sentiment_dict = sentiment_dict
    
    def mutate(self, text: str) -> List[str]:
        """
        Apply mutation to a text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("identity" or "inequality")
        """
        raise NotImplementedError("Subclasses must implement this method")


class SynonymReplacementMutator(MetamorphicMutator):
    """Mutator that replaces words with their synonyms."""
    
    def mutate(self, text: str, n: int = 1) -> List[str]:
        """
        Replace n words in the text with their synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            List of mutated texts
        """
        doc = nlp(text)
        words = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]
        
        if not words:
            return [text]
        
        n = min(n, len(words))
        words_to_replace = random.sample(words, n)
        
        mutated_texts = []
        for word in words_to_replace:
            synonyms = self._get_synonyms(word)
            if synonyms:
                for synonym in synonyms[:3]:  # Limit to 3 synonyms per word
                    mutated_text = text.replace(word, synonym)
                    mutated_texts.append(mutated_text)
        
        return mutated_texts if mutated_texts else [text]
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        
        return synonyms
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("identity")
        """
        return "identity"


class TenseChangeMutator(MetamorphicMutator):
    """Mutator that changes the tense of verbs."""
    
    def mutate(self, text: str) -> List[str]:
        """
        Change the tense of verbs in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        doc = nlp(text)
        verbs = [token for token in doc if token.pos_ == "VERB"]
        
        if not verbs:
            return [text]
        
        mutated_texts = []
        for verb in verbs:
            # Simple past to present
            if verb.tag_ == "VBD":
                present_form = self._get_present_form(verb.text)
                if present_form:
                    mutated_text = text.replace(verb.text, present_form)
                    mutated_texts.append(mutated_text)
            
            # Present to past
            elif verb.tag_ in ["VB", "VBP", "VBZ"]:
                past_form = self._get_past_form(verb.text)
                if past_form:
                    mutated_text = text.replace(verb.text, past_form)
                    mutated_texts.append(mutated_text)
        
        return mutated_texts if mutated_texts else [text]
    
    def _get_present_form(self, verb: str) -> str:
        """
        Get the present form of a verb.
        
        Args:
            verb: Input verb
            
        Returns:
            Present form of the verb
        """
        # Simple rule-based conversion
        if verb.endswith("ed"):
            return verb[:-2]
        return verb
    
    def _get_past_form(self, verb: str) -> str:
        """
        Get the past form of a verb.
        
        Args:
            verb: Input verb
            
        Returns:
            Past form of the verb
        """
        # Simple rule-based conversion
        if verb.endswith("e"):
            return verb + "d"
        return verb + "ed"
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("identity")
        """
        return "identity"


class IntensifierMutator(MetamorphicMutator):
    """Mutator that adds intensifiers to adjectives."""
    
    def mutate(self, text: str) -> List[str]:
        """
        Add intensifiers to adjectives in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        doc = nlp(text)
        adjectives = [token for token in doc if token.pos_ == "ADJ"]
        
        if not adjectives:
            return [text]
        
        mutated_texts = []
        for adj in adjectives:
            for intensifier in INTENSIFIERS:
                # Check if the adjective already has an intensifier
                has_intensifier = False
                for token in doc:
                    if token.pos_ == "ADV" and token.head == adj:
                        has_intensifier = True
                        break
                
                if not has_intensifier:
                    mutated_text = text.replace(adj.text, f"{intensifier} {adj.text}")
                    mutated_texts.append(mutated_text)
        
        return mutated_texts if mutated_texts else [text]
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("inequality")
        """
        return "inequality"


class ReducerMutator(MetamorphicMutator):
    """Mutator that adds reducers to adjectives."""
    
    def mutate(self, text: str) -> List[str]:
        """
        Add reducers to adjectives in the text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        doc = nlp(text)
        adjectives = [token for token in doc if token.pos_ == "ADJ"]
        
        if not adjectives:
            return [text]
        
        mutated_texts = []
        for adj in adjectives:
            for reducer in REDUCERS:
                # Check if the adjective already has a modifier
                has_modifier = False
                for token in doc:
                    if token.pos_ == "ADV" and token.head == adj:
                        has_modifier = True
                        break
                
                if not has_modifier:
                    mutated_text = text.replace(adj.text, f"{reducer} {adj.text}")
                    mutated_texts.append(mutated_text)
        
        return mutated_texts if mutated_texts else [text]
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("inequality")
        """
        return "inequality"


class PositivePhraseMutator(MetamorphicMutator):
    """Mutator that adds positive phrases to the text."""
    
    def mutate(self, text: str) -> List[str]:
        """
        Add positive phrases to the text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        mutated_texts = []
        for phrase in POSITIVE_PHRASES:
            mutated_text = f"{text} {phrase}"
            mutated_texts.append(mutated_text)
        
        return mutated_texts
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("inequality")
        """
        return "inequality"


class NegativePhraseMutator(MetamorphicMutator):
    """Mutator that adds negative phrases to the text."""
    
    def mutate(self, text: str) -> List[str]:
        """
        Add negative phrases to the text.
        
        Args:
            text: Input text
            
        Returns:
            List of mutated texts
        """
        mutated_texts = []
        for phrase in NEGATIVE_PHRASES:
            mutated_text = f"{text} {phrase}"
            mutated_texts.append(mutated_text)
        
        return mutated_texts
    
    def get_relation_type(self) -> str:
        """
        Get the relation type of the mutator.
        
        Returns:
            Relation type ("inequality")
        """
        return "inequality"
