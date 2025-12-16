#!/usr/bin/env python3
"""
Example Usage of Matt AI

This script demonstrates basic usage of the Matt AI system.
"""

import logging
from src.matt_ai import EthicalController, SelfTrainingLLM, IterativeTrainer
from src.matt_ai.data_utils import DataPreparer


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_ethical_controller():
    """Demonstrate the ethical controller."""
    print("\n" + "=" * 60)
    print("Example 1: Ethical Standards Controller")
    print("=" * 60)
    
    # Initialize ethical controller
    ethical_controller = EthicalController()
    
    # Display core principles
    print("\nCore Ethical Principles:")
    for i, principle in enumerate(ethical_controller.get_core_principles(), 1):
        print(f"  {i}. {principle}")
    
    # Test action validation
    print("\n--- Testing Action Validation ---")
    
    # Valid action
    is_valid, reason = ethical_controller.validate_action(
        "Generate educational content about machine learning"
    )
    print(f"Action: Generate educational content")
    print(f"Valid: {is_valid}, Reason: {reason}")
    
    # Invalid action
    is_valid, reason = ethical_controller.validate_action(
        "Generate content that promotes violence"
    )
    print(f"\nAction: Generate violent content")
    print(f"Valid: {is_valid}, Reason: {reason}")


def example_text_generation():
    """Demonstrate text generation."""
    print("\n" + "=" * 60)
    print("Example 2: Text Generation with Ethical Validation")
    print("=" * 60)
    
    # Initialize components
    ethical_controller = EthicalController()
    model = SelfTrainingLLM(
        model_name="gpt2",
        device="cpu",  # Use CPU for demo
        ethical_controller=ethical_controller
    )
    
    # Generate text
    prompt = "The future of artificial intelligence"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating...")
    
    generated_texts = model.generate(
        prompt,
        max_length=80,
        temperature=0.7,
        num_return_sequences=2
    )
    
    print("\nGenerated Texts:")
    for i, text in enumerate(generated_texts, 1):
        print(f"\n{i}. {text}")


def example_data_preparation():
    """Demonstrate data preparation."""
    print("\n" + "=" * 60)
    print("Example 3: Data Preparation")
    print("=" * 60)
    
    # Initialize data preparer
    data_preparer = DataPreparer()
    
    # Load sample data
    print("\nLoading sample data...")
    texts = data_preparer.load_sample_data()
    print(f"Loaded {len(texts)} sample texts")
    
    # Clean and prepare data
    print("\nPreparing training data...")
    train_texts, val_texts = data_preparer.prepare_training_data(texts)
    print(f"Training texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    
    # Show sample
    print("\nSample training text:")
    print(train_texts[0][:200] + "...")


def example_mini_training():
    """Demonstrate a minimal training example."""
    print("\n" + "=" * 60)
    print("Example 4: Mini Training Session")
    print("=" * 60)
    
    # Initialize components
    ethical_controller = EthicalController()
    model = SelfTrainingLLM(
        model_name="gpt2",
        device="cpu",
        ethical_controller=ethical_controller
    )
    
    # Prepare minimal data
    data_preparer = DataPreparer()
    texts = data_preparer.load_sample_data()[:10]  # Use only 10 texts for speed
    train_texts, val_texts = data_preparer.prepare_training_data(texts)
    
    # Initialize trainer
    trainer = IterativeTrainer(
        model=model,
        ethical_controller=ethical_controller,
        learning_rate=5e-5,
        batch_size=2,  # Small batch for demo
        max_iterations=5  # Very few iterations for demo
    )
    
    print("\nStarting mini training session...")
    print("(This is a minimal demo - real training requires more data and time)")
    
    # Train for just 1 epoch with few steps
    results = trainer.train(
        training_texts=train_texts[:4],  # Only 4 texts
        validation_texts=val_texts[:2],  # Only 2 validation texts
        num_epochs=1,
        save_every=100,  # Don't save during demo
        evaluate_every=2
    )
    
    print("\nTraining completed!")
    print(f"Iterations: {results['iterations_completed']}")
    print(f"Duration: {results['session_duration']}")


def example_model_info():
    """Display model information."""
    print("\n" + "=" * 60)
    print("Example 5: Model Information")
    print("=" * 60)
    
    # Initialize model
    model = SelfTrainingLLM(model_name="gpt2", device="cpu")
    
    # Get and display info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def main():
    """Run all examples."""
    setup_logging()
    
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  Matt AI - Example Usage Demonstrations".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    try:
        # Run examples
        example_ethical_controller()
        input("\nPress Enter to continue to next example...")
        
        example_data_preparation()
        input("\nPress Enter to continue to next example...")
        
        example_model_info()
        input("\nPress Enter to continue to next example...")
        
        # Text generation example (requires model download)
        response = input("\nRun text generation example? This will download GPT-2 model (~500MB). (y/n): ")
        if response.lower() == 'y':
            example_text_generation()
        
        # Training example
        response = input("\nRun mini training example? This may take a few minutes. (y/n): ")
        if response.lower() == 'y':
            example_mini_training()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Try: python main.py info")
        print("  2. Try: python main.py generate --prompt 'Your prompt here'")
        print("  3. Try: python main.py train --epochs 3")
        print("\nSee README.md for full documentation.")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError during examples: {e}")
        logging.exception("Example error")


if __name__ == '__main__':
    main()
