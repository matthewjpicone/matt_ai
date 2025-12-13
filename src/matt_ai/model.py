"""
Self-Training Large Language Model

Core model architecture based on transformer technology with self-improvement capabilities.
"""

import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Config,
)


class SelfTrainingLLM:
    """
    A self-training large language model with iterative improvement capabilities.
    
    This model starts with a base architecture and progressively improves
    through self-supervised learning on English text data.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        ethical_controller=None
    ):
        """
        Initialize the self-training LLM.
        
        Args:
            model_name: Base model to use (default: gpt2)
            device: Device to run model on (cpu/cuda)
            ethical_controller: EthicalController instance for compliance
        """
        self.logger = logging.getLogger(__name__)
        self.ethical_controller = ethical_controller
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialize_model()
        
        # Training metrics
        self.training_history = {
            "iterations": 0,
            "total_loss": 0.0,
            "improvements": []
        }
    
    def _initialize_model(self):
        """Initialize or load the language model."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            self.logger.info(f"Model loaded successfully with {self._count_parameters()} parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Creating new model from scratch")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new model from scratch if loading fails."""
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        
        self.model = GPT2LMHeadModel(config)
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        # Validate with ethical controller if available
        if self.ethical_controller:
            is_valid, reason = self.ethical_controller.validate_action(
                f"generate text from prompt: {prompt[:50]}..."
            )
            if not is_valid:
                self.logger.warning(f"Generation blocked: {reason}")
                return [f"[Generation blocked: {reason}]"]
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        # Validate outputs
        if self.ethical_controller:
            validated_texts = []
            for text in generated_texts:
                is_valid, reason = self.ethical_controller.validate_output(text)
                if is_valid:
                    validated_texts.append(text)
                else:
                    self.logger.warning(f"Output blocked: {reason}")
                    validated_texts.append(f"[Output blocked: {reason}]")
            return validated_texts
        
        return generated_texts
    
    def train_step(
        self,
        texts: List[str],
        learning_rate: float = 5e-5
    ) -> float:
        """
        Perform a single training step on a batch of texts.
        
        Args:
            texts: List of training texts
            learning_rate: Learning rate for this step
            
        Returns:
            Training loss
        """
        self.model.train()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update weights (simplified - actual optimizer should be external)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
                    param.grad.zero_()
        
        loss_value = loss.item()
        self.training_history["total_loss"] += loss_value
        self.training_history["iterations"] += 1
        
        return loss_value
    
    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """
        Evaluate model on a set of texts.
        
        Args:
            texts: List of evaluation texts
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for text in texts:
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(texts) if texts else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_samples": len(texts)
        }
    
    def save(self, save_path: str):
        """Save model and tokenizer to disk."""
        self.logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training history
        history_path = f"{save_path}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load(self, load_path: str):
        """Load model and tokenizer from disk."""
        self.logger.info(f"Loading model from {load_path}")
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        # Load training history if available
        history_path = f"{load_path}/training_history.json"
        try:
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        except FileNotFoundError:
            self.logger.warning("No training history found")
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": self._count_parameters(),
            "training_iterations": self.training_history["iterations"],
            "total_loss": self.training_history["total_loss"],
            "avg_loss": (
                self.training_history["total_loss"] / self.training_history["iterations"]
                if self.training_history["iterations"] > 0 else 0.0
            )
        }
