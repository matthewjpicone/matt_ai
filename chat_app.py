#!/usr/bin/env python3
"""
Matt AI - Chat Application with Web Scraping

Interactive chat interface with autonomous web scraping for training data.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.matt_ai import EthicalController, SelfTrainingLLM, IterativeTrainer
from src.matt_ai.web_scraper import WebScraper
from src.matt_ai.instruction_handler import InstructionHandler
from src.matt_ai.chat_interface import ChatInterface


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'chat_app.log')
        ]
    )


def main():
    """Main entry point for the chat application."""
    parser = argparse.ArgumentParser(
        description="Matt AI - Interactive Chat with Web Scraping and Learning"
    )
    
    parser.add_argument(
        '--model-name',
        default='gpt2',
        help='Base model name (default: gpt2)'
    )
    parser.add_argument(
        '--model-path',
        help='Path to pre-trained model (optional)'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='Device to use (cpu/cuda)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=2.0,
        help='Rate limit delay for web scraping in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--max-pages-per-domain',
        type=int,
        default=10,
        help='Maximum pages to scrape per domain (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (default: 4)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate (default: 5e-5)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=7860,
        help='Server port (default: 7860)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("Matt AI - Chat Application")
    logger.info("=" * 50)
    
    try:
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
        
        # Load pre-trained model if specified
        if args.model_path:
            logger.info(f"Loading model from: {args.model_path}")
            model.load(args.model_path)
        
        # Initialize web scraper
        logger.info("Initializing web scraper...")
        web_scraper = WebScraper(
            rate_limit_delay=args.rate_limit,
            max_pages_per_domain=args.max_pages_per_domain
        )
        
        # Initialize instruction handler
        logger.info("Initializing instruction handler...")
        instruction_handler = InstructionHandler(
            instruction_file="data/instructions.json"
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = IterativeTrainer(
            model=model,
            ethical_controller=ethical_controller,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            checkpoint_dir="checkpoints"
        )
        
        # Create chat interface
        logger.info("Creating chat interface...")
        chat_interface = ChatInterface(
            model=model,
            web_scraper=web_scraper,
            instruction_handler=instruction_handler,
            trainer=trainer
        )
        
        # Launch interface
        logger.info(f"Launching interface on port {args.server_port}...")
        logger.info("=" * 50)
        logger.info("Chat interface is ready!")
        logger.info(f"Open your browser and navigate to: http://localhost:{args.server_port}")
        if args.share:
            logger.info("Public sharing link will be generated...")
        logger.info("=" * 50)
        
        chat_interface.launch(
            share=args.share,
            server_port=args.server_port,
            server_name="0.0.0.0"
        )
        
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
