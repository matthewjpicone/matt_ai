"""
Data utilities for preparing and managing training data.

Focuses on English language text processing.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple


class DataPreparer:
    """
    Prepare and process English text data for training.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_sample_data(self) -> List[str]:
        """
        Load sample English text data for initial training.
        
        Returns:
            List of English text samples
        """
        # Sample English texts for demonstration
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typography.",
            "Artificial intelligence is transforming the way we interact with technology and process information.",
            "Language models learn patterns from text data to generate coherent and contextually relevant responses.",
            "Machine learning algorithms can identify complex patterns in large datasets through iterative training.",
            "Natural language processing enables computers to understand and generate human language effectively.",
            "The development of ethical AI systems requires careful consideration of societal impacts and values.",
            "Deep learning models use neural networks with multiple layers to learn hierarchical representations.",
            "Text generation models can produce creative content while maintaining grammatical and semantic coherence.",
            "Self-supervised learning allows models to improve without explicit human-labeled training data.",
            "The architecture of transformer models has revolutionized natural language understanding tasks.",
            "Training large language models requires significant computational resources and careful optimization.",
            "Iterative improvement processes enable AI systems to progressively enhance their capabilities.",
            "English language processing involves understanding grammar, semantics, and contextual nuances.",
            "Automated text generation must balance creativity with factual accuracy and ethical considerations.",
            "The quality of training data directly impacts the performance and reliability of language models.",
            "Modern AI systems can process and generate text at scales previously unimaginable to researchers.",
            "Understanding language requires not just vocabulary but also knowledge of pragmatics and discourse.",
            "Continuous learning mechanisms allow AI models to adapt to new information and changing contexts.",
            "The evaluation of language models involves multiple metrics including coherence and relevance.",
            "Responsible AI development prioritizes transparency, fairness, and accountability in all applications.",
        ]
        
        return sample_texts
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize English text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def prepare_training_data(
        self,
        texts: List[str],
        min_length: int = 20,
        max_length: int = 1000
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare training data by cleaning and splitting into train/validation.
        
        Args:
            texts: Raw texts
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Tuple of (training_texts, validation_texts)
        """
        cleaned_texts = []
        
        for text in texts:
            # Clean text
            cleaned = self.clean_text(text)
            
            # Filter by length
            if min_length <= len(cleaned) <= max_length:
                cleaned_texts.append(cleaned)
        
        # Split into train (80%) and validation (20%)
        split_idx = int(len(cleaned_texts) * 0.8)
        
        training_texts = cleaned_texts[:split_idx]
        validation_texts = cleaned_texts[split_idx:]
        
        self.logger.info(f"Prepared {len(training_texts)} training and {len(validation_texts)} validation texts")
        
        return training_texts, validation_texts
    
    def augment_data(self, texts: List[str], augmentation_factor: int = 2) -> List[str]:
        """
        Augment training data with variations.
        
        Args:
            texts: Original texts
            augmentation_factor: How many variations to create
            
        Returns:
            Augmented text list
        """
        augmented = texts.copy()
        
        for text in texts[:len(texts) // augmentation_factor]:
            # Simple augmentation: sentence reordering for multi-sentence texts
            sentences = self.split_into_sentences(text)
            if len(sentences) > 1:
                # Reverse sentence order as a simple augmentation
                augmented.append(' '.join(reversed(sentences)))
        
        self.logger.info(f"Augmented data from {len(texts)} to {len(augmented)} texts")
        
        return augmented
    
    def load_from_file(self, file_path: str) -> List[str]:
        """
        Load text data from a file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of texts
        """
        path = Path(file_path)
        
        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (paragraphs or sentences)
            if '\n\n' in content:
                texts = content.split('\n\n')
            else:
                texts = self.split_into_sentences(content)
            
            self.logger.info(f"Loaded {len(texts)} texts from {file_path}")
            return texts
            
        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            return []
    
    def save_to_file(self, texts: List[str], file_path: str):
        """
        Save texts to a file.
        
        Args:
            texts: List of texts to save
            file_path: Output file path
        """
        path = Path(file_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n\n')
            
            self.logger.info(f"Saved {len(texts)} texts to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
    
    def get_sample_prompts(self) -> List[str]:
        """
        Get sample prompts for text generation testing.
        
        Returns:
            List of English prompts
        """
        return [
            "In the field of artificial intelligence,",
            "The importance of ethical considerations in AI",
            "Machine learning models can",
            "The future of technology will",
            "Natural language processing enables",
            "When training neural networks,",
            "The benefits of self-improving systems include",
            "Language understanding requires",
            "Advanced AI systems should",
            "The key to responsible AI development is",
        ]
