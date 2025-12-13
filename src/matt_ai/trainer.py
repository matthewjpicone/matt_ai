"""
Iterative Trainer for Self-Improving LLM

Implements the training loop with self-improvement mechanisms.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


class IterativeTrainer:
    """
    Iterative trainer for self-improving language model.
    
    This trainer implements a self-training loop where the model
    progressively improves through iterative learning cycles.
    """
    
    def __init__(
        self,
        model,
        ethical_controller=None,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        max_iterations: int = 100,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the iterative trainer.
        
        Args:
            model: SelfTrainingLLM instance
            ethical_controller: EthicalController for compliance
            learning_rate: Initial learning rate
            batch_size: Training batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            max_iterations: Maximum training iterations per session
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.ethical_controller = ethical_controller
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_iterations = max_iterations
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_iterations
        )
        
        # Training state
        self.current_iteration = 0
        self.session_start = None
        self.training_metrics = {
            "losses": [],
            "perplexities": [],
            "learning_rates": [],
            "improvements": []
        }
    
    def train(
        self,
        training_texts: List[str],
        validation_texts: Optional[List[str]] = None,
        num_epochs: int = 1,
        save_every: int = 10,
        evaluate_every: int = 5
    ) -> Dict:
        """
        Train the model iteratively with self-improvement.
        
        Args:
            training_texts: List of texts for training
            validation_texts: Optional list of texts for validation
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N iterations
            evaluate_every: Evaluate every N iterations
            
        Returns:
            Dictionary of training results
        """
        self.session_start = datetime.now()
        self.logger.info(f"Starting training session at {self.session_start}")
        self.logger.info(f"Training on {len(training_texts)} texts for {num_epochs} epochs")
        
        # Apply ethical guidelines
        if self.ethical_controller:
            guidelines = self.ethical_controller.get_learning_guidelines()
            self.logger.info(f"Training with ethical guidelines: {guidelines}")
        
        # Create dataset and dataloader
        dataset = TextDataset(training_texts, self.model.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Training loop
        self.model.model.train()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Check ethical compliance
                if self.ethical_controller:
                    if not self.ethical_controller.check_training_compliance(
                        self.current_iteration,
                        self.session_start
                    ):
                        self.logger.warning("Training stopped due to ethical constraints")
                        return self._get_training_results()
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                
                # Forward pass
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Update weights after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.current_iteration += 1
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
                })
                
                # Save checkpoint
                if self.current_iteration % save_every == 0:
                    self._save_checkpoint(f"iter_{self.current_iteration}")
                
                # Evaluate
                if validation_texts and self.current_iteration % evaluate_every == 0:
                    eval_metrics = self._evaluate(validation_texts)
                    self.logger.info(f"Evaluation at iteration {self.current_iteration}: {eval_metrics}")
                    
                    # Check for improvement
                    if eval_metrics["loss"] < best_loss:
                        improvement = best_loss - eval_metrics["loss"]
                        best_loss = eval_metrics["loss"]
                        self.training_metrics["improvements"].append({
                            "iteration": self.current_iteration,
                            "improvement": improvement,
                            "new_loss": best_loss
                        })
                        self.logger.info(f"Model improved! New best loss: {best_loss:.4f}")
                        self._save_checkpoint("best_model")
            
            # Record epoch metrics
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.training_metrics["losses"].append(avg_epoch_loss)
            self.training_metrics["learning_rates"].append(self.scheduler.get_last_lr()[0])
            
            self.logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        return self._get_training_results()
    
    def _evaluate(self, validation_texts: List[str]) -> Dict:
        """Evaluate the model on validation data."""
        self.model.model.eval()
        
        with torch.no_grad():
            eval_results = self.model.evaluate(validation_texts[:100])  # Sample for speed
            self.training_metrics["perplexities"].append(eval_results["perplexity"])
        
        self.model.model.train()
        return eval_results
    
    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        
        self.model.save(str(checkpoint_path))
        
        # Save optimizer state
        torch.save({
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "iteration": self.current_iteration,
            "training_metrics": self.training_metrics
        }, checkpoint_path / "trainer_state.pt")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _get_training_results(self) -> Dict:
        """Get training results summary."""
        return {
            "iterations_completed": self.current_iteration,
            "session_duration": str(datetime.now() - self.session_start) if self.session_start else "N/A",
            "final_loss": self.training_metrics["losses"][-1] if self.training_metrics["losses"] else None,
            "best_perplexity": min(self.training_metrics["perplexities"]) if self.training_metrics["perplexities"] else None,
            "num_improvements": len(self.training_metrics["improvements"]),
            "improvements": self.training_metrics["improvements"],
            "ethical_violations": (
                len(self.ethical_controller.get_violation_log())
                if self.ethical_controller else 0
            )
        }
    
    def self_improve(
        self,
        seed_texts: List[str],
        num_cycles: int = 5,
        generation_per_cycle: int = 10
    ) -> Dict:
        """
        Implement self-improvement cycle.
        
        The model generates new training data from seed texts,
        evaluates quality, and trains on high-quality generations.
        
        Args:
            seed_texts: Initial seed texts for generation
            num_cycles: Number of self-improvement cycles
            generation_per_cycle: Number of texts to generate per cycle
            
        Returns:
            Results of self-improvement process
        """
        self.logger.info(f"Starting self-improvement with {num_cycles} cycles")
        
        improvement_results = []
        
        for cycle in range(num_cycles):
            self.logger.info(f"Self-improvement cycle {cycle + 1}/{num_cycles}")
            
            # Generate new training data
            generated_texts = []
            for seed_text in seed_texts[:generation_per_cycle]:
                generated = self.model.generate(
                    seed_text,
                    max_length=200,
                    temperature=0.8,
                    num_return_sequences=1
                )
                generated_texts.extend(generated)
            
            # Evaluate quality of generated texts
            quality_scores = self._evaluate_generation_quality(generated_texts)
            
            # Filter high-quality generations
            threshold = 0.6  # Quality threshold
            high_quality_texts = [
                text for text, score in zip(generated_texts, quality_scores)
                if score > threshold
            ]
            
            if high_quality_texts:
                self.logger.info(f"Generated {len(high_quality_texts)} high-quality texts")
                
                # Train on high-quality generations
                cycle_results = self.train(
                    high_quality_texts,
                    validation_texts=seed_texts,
                    num_epochs=1,
                    save_every=1000,  # Don't save too often
                    evaluate_every=len(high_quality_texts) // self.batch_size + 1
                )
                
                improvement_results.append({
                    "cycle": cycle + 1,
                    "texts_generated": len(generated_texts),
                    "high_quality_texts": len(high_quality_texts),
                    "results": cycle_results
                })
            else:
                self.logger.warning(f"No high-quality texts generated in cycle {cycle + 1}")
        
        return {
            "cycles_completed": num_cycles,
            "cycle_results": improvement_results
        }
    
    def _evaluate_generation_quality(self, texts: List[str]) -> List[float]:
        """
        Evaluate quality of generated texts.
        
        Simple heuristic-based quality evaluation.
        In production, this would use more sophisticated metrics.
        """
        quality_scores = []
        
        for text in texts:
            score = 0.0
            
            # Length check (not too short, not too long)
            if 50 < len(text) < 500:
                score += 0.3
            
            # Check for complete sentences
            if text.strip().endswith(('.', '!', '?')):
                score += 0.2
            
            # Check for reasonable word count
            words = text.split()
            if 10 < len(words) < 100:
                score += 0.3
            
            # Check for diversity (not too repetitive)
            unique_words = len(set(words))
            if len(words) > 0 and unique_words / len(words) > 0.5:
                score += 0.2
            
            quality_scores.append(min(score, 1.0))
        
        return quality_scores
