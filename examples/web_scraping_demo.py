#!/usr/bin/env python3
"""
Web Scraping Demo

Demonstrates how to use the web scraper programmatically.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matt_ai.web_scraper import WebScraper
from src.matt_ai.data_utils import DataPreparer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run web scraping demo."""
    logger.info("=" * 50)
    logger.info("Web Scraping Demo")
    logger.info("=" * 50)
    
    # Initialize web scraper with ethical settings
    scraper = WebScraper(
        rate_limit_delay=2.0,  # 2 seconds between requests
        max_pages_per_domain=5,  # Max 5 pages per domain
        timeout=10
    )
    
    # Example URLs to scrape (Wikipedia articles about AI)
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
    ]
    
    logger.info(f"Starting to scrape {len(urls)} URLs...")
    logger.info("Note: This demo respects robots.txt and implements rate limiting")
    
    # Scrape URLs
    texts = scraper.scrape_urls(urls, max_pages=10)
    
    logger.info(f"\nScraping completed!")
    logger.info(f"Collected {len(texts)} texts")
    
    # Display statistics
    stats = scraper.get_statistics()
    logger.info("\n--- Scraping Statistics ---")
    logger.info(f"Pages scraped: {stats['pages_scraped']}")
    logger.info(f"Pages failed: {stats['pages_failed']}")
    logger.info(f"Pages blocked: {stats['pages_blocked']}")
    logger.info(f"Unique domains: {stats['unique_domains']}")
    logger.info(f"Total text collected: {stats['total_text_collected']:,} characters")
    
    # Show sample of collected text
    if texts:
        logger.info("\n--- Sample Text (first 500 chars) ---")
        logger.info(texts[0][:500] + "...")
        
        # Prepare data for training
        data_preparer = DataPreparer()
        train_texts, val_texts = data_preparer.prepare_training_data(texts)
        
        logger.info(f"\n--- Prepared Training Data ---")
        logger.info(f"Training texts: {len(train_texts)}")
        logger.info(f"Validation texts: {len(val_texts)}")
        
        # Optionally save to file
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        data_preparer.save_to_file(train_texts, "data/scraped_train.txt")
        data_preparer.save_to_file(val_texts, "data/scraped_val.txt")
        
        logger.info(f"\nData saved to data/ directory")
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo completed!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
