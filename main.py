#!/usr/bin/env python3
"""
Matt AI - Main Entry Point

Self-training large language model system with ethical standards.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.matt_ai import EthicalController, SelfTrainingLLM, IterativeTrainer
from src.matt_ai.data_utils import DataPreparer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('matt_ai.log')
        ]
    )


def train_mode(args):
    """Run training mode."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Matt AI - Training Mode")
    logger.info("=" * 50)
    
    # Initialize ethical controller
    logger.info("Initializing ethical standards controller...")
    ethical_controller = EthicalController()
    
    # Display core principles
    principles = ethical_controller.get_core_principles()
    logger.info("Core Ethical Principles:")
    for i, principle in enumerate(principles, 1):
        logger.info(f"  {i}. {principle}")
    
    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")
    model = SelfTrainingLLM(
        model_name=args.model_name,
        device=args.device,
        ethical_controller=ethical_controller
    )
    
    # Load or prepare data
    logger.info("Preparing training data...")
    data_preparer = DataPreparer()
    
    if args.data_file:
        texts = data_preparer.load_from_file(args.data_file)
    else:
        logger.info("Using sample data for demonstration")
        texts = data_preparer.load_sample_data()
        texts = data_preparer.augment_data(texts, augmentation_factor=3)
    
    train_texts, val_texts = data_preparer.prepare_training_data(texts)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = IterativeTrainer(
        model=model,
        ethical_controller=ethical_controller,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    results = trainer.train(
        training_texts=train_texts,
        validation_texts=val_texts,
        num_epochs=args.epochs,
        save_every=args.save_every,
        evaluate_every=args.eval_every
    )
    
    logger.info("Training completed!")
    logger.info(f"Results: {results}")
    
    # Save final model
    if args.output_dir:
        logger.info(f"Saving final model to {args.output_dir}")
        model.save(args.output_dir)
    
    # Display ethical violations if any
    violations = ethical_controller.get_violation_log()
    if violations:
        logger.warning(f"Ethical violations detected: {len(violations)}")
        for v in violations:
            logger.warning(f"  - {v['reason']}")
    else:
        logger.info("No ethical violations detected during training")


def generate_mode(args):
    """Run text generation mode."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Matt AI - Generation Mode")
    logger.info("=" * 50)
    
    # Initialize ethical controller
    ethical_controller = EthicalController()
    
    # Initialize model
    logger.info(f"Loading model from: {args.model_path or args.model_name}")
    model = SelfTrainingLLM(
        model_name=args.model_name,
        device=args.device,
        ethical_controller=ethical_controller
    )
    
    if args.model_path:
        model.load(args.model_path)
    
    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        data_preparer = DataPreparer()
        prompts = data_preparer.get_sample_prompts()
        logger.info("Using sample prompts for demonstration")
    
    # Generate for each prompt
    logger.info(f"Generating text for {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n--- Prompt {i} ---")
        logger.info(f"Input: {prompt}")
        
        generated = model.generate(
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        for j, text in enumerate(generated, 1):
            logger.info(f"Generation {j}:")
            logger.info(text)
            logger.info("")


def self_improve_mode(args):
    """Run self-improvement mode."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Matt AI - Self-Improvement Mode")
    logger.info("=" * 50)
    
    # Initialize ethical controller
    ethical_controller = EthicalController()
    
    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")
    model = SelfTrainingLLM(
        model_name=args.model_name,
        device=args.device,
        ethical_controller=ethical_controller
    )
    
    if args.model_path:
        logger.info(f"Loading model from: {args.model_path}")
        model.load(args.model_path)
    
    # Prepare seed data
    data_preparer = DataPreparer()
    seed_texts = data_preparer.load_sample_data()
    
    # Initialize trainer
    trainer = IterativeTrainer(
        model=model,
        ethical_controller=ethical_controller,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Run self-improvement
    logger.info(f"Starting self-improvement with {args.cycles} cycles...")
    results = trainer.self_improve(
        seed_texts=seed_texts,
        num_cycles=args.cycles,
        generation_per_cycle=args.generations_per_cycle
    )
    
    logger.info("Self-improvement completed!")
    logger.info(f"Results: {results}")
    
    # Save improved model
    if args.output_dir:
        logger.info(f"Saving improved model to {args.output_dir}")
        model.save(args.output_dir)


def info_mode(args):
    """Display information about the system."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Matt AI - System Information")
    logger.info("=" * 50)
    
    # Load ethical standards
    ethical_controller = EthicalController()
    
    logger.info("\n--- Core Ethical Principles ---")
    for i, principle in enumerate(ethical_controller.get_core_principles(), 1):
        logger.info(f"{i}. {principle}")
    
    logger.info("\n--- Learning Guidelines ---")
    guidelines = ethical_controller.get_learning_guidelines()
    for key, value in guidelines.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n--- Operational Constraints ---")
    constraints = ethical_controller.get_operational_constraints()
    for key, value in constraints.items():
        logger.info(f"{key}: {value}")
    
    # Model info if path provided
    if args.model_path:
        logger.info(f"\n--- Model Information ---")
        model = SelfTrainingLLM(model_name="gpt2", device=args.device)
        model.load(args.model_path)
        info = model.get_model_info()
        for key, value in info.items():
            logger.info(f"{key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Matt AI - Self-Training Large Language Model with Ethical Standards"
    )
    
    # Global arguments
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--device', default=None, help='Device to use (cpu/cuda)')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model-name', default='gpt2', help='Base model name')
    train_parser.add_argument('--data-file', help='Path to training data file')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    train_parser.add_argument('--max-iterations', type=int, default=100, help='Max iterations per session')
    train_parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N iterations')
    train_parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N iterations')
    train_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--output-dir', help='Final model output directory')
    
    # Generate mode
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--model-name', default='gpt2', help='Base model name')
    gen_parser.add_argument('--model-path', help='Path to trained model')
    gen_parser.add_argument('--prompt', help='Input prompt')
    gen_parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling parameter')
    gen_parser.add_argument('--num-sequences', type=int, default=1, help='Number of sequences to generate')
    
    # Self-improve mode
    improve_parser = subparsers.add_parser('self-improve', help='Self-improvement mode')
    improve_parser.add_argument('--model-name', default='gpt2', help='Base model name')
    improve_parser.add_argument('--model-path', help='Path to model to improve')
    improve_parser.add_argument('--cycles', type=int, default=3, help='Number of improvement cycles')
    improve_parser.add_argument('--generations-per-cycle', type=int, default=10, help='Generations per cycle')
    improve_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    improve_parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    improve_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    improve_parser.add_argument('--output-dir', help='Output directory for improved model')
    
    # Info mode
    info_parser = subparsers.add_parser('info', help='Display system information')
    info_parser.add_argument('--model-path', help='Path to model for info')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Route to appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'generate':
        generate_mode(args)
    elif args.mode == 'self-improve':
        self_improve_mode(args)
    elif args.mode == 'info':
        info_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
