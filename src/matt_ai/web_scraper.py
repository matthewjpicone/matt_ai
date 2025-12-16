"""
Web Scraper Module for Training Data Collection

Implements ethical web scraping with:
- robots.txt compliance
- Rate limiting
- Error handling and logging
"""

import logging
import time
import requests
from typing import List, Optional, Set, Dict
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from collections import deque
import re


class WebScraper:
    """
    Ethical web scraper for collecting training data.
    
    Respects robots.txt, implements rate limiting, and handles errors gracefully.
    """
    
    def __init__(
        self,
        rate_limit_delay: float = 2.0,
        max_pages_per_domain: int = 10,
        timeout: int = 10,
        user_agent: str = "MattAI-Trainer/1.0 (Educational Bot)"
    ):
        """
        Initialize the web scraper.
        
        Args:
            rate_limit_delay: Delay between requests in seconds
            max_pages_per_domain: Maximum pages to scrape per domain
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = rate_limit_delay
        self.max_pages_per_domain = max_pages_per_domain
        self.timeout = timeout
        self.user_agent = user_agent
        
        # Track visited URLs and domain statistics
        self.visited_urls: Set[str] = set()
        self.domain_counts: Dict[str, int] = {}
        self.robot_parsers: Dict[str, RobotFileParser] = {}
        
        # Statistics
        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "pages_blocked": 0,
            "total_text_collected": 0,
            "errors": []
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def _get_robot_parser(self, url: str) -> Optional[RobotFileParser]:
        """
        Get or create robot parser for a domain.
        
        Args:
            url: URL to get parser for
            
        Returns:
            RobotFileParser instance or None if unavailable
        """
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain in self.robot_parsers:
            return self.robot_parsers[domain]
        
        # Create new parser
        robots_url = urljoin(domain, '/robots.txt')
        parser = RobotFileParser()
        parser.set_url(robots_url)
        
        try:
            parser.read()
            self.robot_parsers[domain] = parser
            self.logger.info(f"Loaded robots.txt for {domain}")
            return parser
        except Exception as e:
            self.logger.warning(f"Could not load robots.txt for {domain}: {e}")
            # Create permissive parser as fallback
            self.robot_parsers[domain] = None
            return None
    
    def _can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed to fetch, False otherwise
        """
        parser = self._get_robot_parser(url)
        
        if parser is None:
            # If robots.txt not available, be cautious and limit
            parsed = urlparse(url)
            domain = parsed.netloc
            return self.domain_counts.get(domain, 0) < self.max_pages_per_domain
        
        return parser.can_fetch(self.user_agent, url)
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _should_scrape_domain(self, url: str) -> bool:
        """
        Check if we should continue scraping this domain.
        
        Args:
            url: URL to check
            
        Returns:
            True if should scrape, False if domain limit reached
        """
        domain = self._get_domain(url)
        return self.domain_counts.get(domain, 0) < self.max_pages_per_domain
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            Cleaned text content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract links from page.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            # Filter out non-HTTP(S) links and fragments
            if absolute_url.startswith(('http://', 'https://')) and '#' not in absolute_url:
                links.append(absolute_url)
        
        return links
    
    def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape a single URL and extract text content.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        # Check if already visited
        if url in self.visited_urls:
            return None
        
        # Check robots.txt
        if not self._can_fetch(url):
            self.logger.info(f"Blocked by robots.txt: {url}")
            self.stats["pages_blocked"] += 1
            return None
        
        # Check domain limit
        if not self._should_scrape_domain(url):
            self.logger.info(f"Domain limit reached for: {self._get_domain(url)}")
            return None
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Fetch page
            self.logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text
            text = self._extract_text(soup)
            
            # Update tracking
            self.visited_urls.add(url)
            domain = self._get_domain(url)
            self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
            
            # Update statistics
            self.stats["pages_scraped"] += 1
            self.stats["total_text_collected"] += len(text)
            
            self.logger.info(f"Successfully scraped {len(text)} characters from {url}")
            
            return text
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            self.stats["pages_failed"] += 1
            self.stats["errors"].append({"url": url, "error": str(e)})
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error scraping {url}: {e}")
            self.stats["pages_failed"] += 1
            self.stats["errors"].append({"url": url, "error": str(e)})
            return None
    
    def scrape_urls(self, urls: List[str], max_pages: int = 50) -> List[str]:
        """
        Scrape multiple URLs and collect text content.
        
        Args:
            urls: List of URLs to scrape
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of extracted text content
        """
        texts = []
        scraped_count = 0
        
        for url in urls:
            if scraped_count >= max_pages:
                self.logger.info(f"Reached maximum page limit: {max_pages}")
                break
            
            text = self.scrape_url(url)
            if text and len(text) > 100:  # Only keep substantial content
                texts.append(text)
                scraped_count += 1
        
        self.logger.info(f"Scraping completed. Collected {len(texts)} texts")
        return texts
    
    def crawl_website(
        self,
        start_url: str,
        max_pages: int = 20,
        same_domain_only: bool = True
    ) -> List[str]:
        """
        Crawl a website starting from a URL.
        
        Args:
            start_url: Starting URL for crawl
            max_pages: Maximum pages to crawl
            same_domain_only: Only follow links within same domain
            
        Returns:
            List of extracted text content
        """
        texts = []
        to_visit = deque([start_url])
        start_domain = self._get_domain(start_url)
        
        while to_visit and len(texts) < max_pages:
            url = to_visit.popleft()
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            # Check domain restriction
            if same_domain_only and self._get_domain(url) != start_domain:
                continue
            
            # Scrape the page
            try:
                time.sleep(self.rate_limit_delay)
                
                if not self._can_fetch(url):
                    self.stats["pages_blocked"] += 1
                    continue
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text
                text = self._extract_text(soup)
                if text and len(text) > 100:
                    texts.append(text)
                    self.logger.info(f"Crawled {len(texts)}/{max_pages}: {url}")
                
                # Extract and queue new links
                if len(texts) < max_pages:
                    links = self._extract_links(soup, url)
                    for link in links:
                        if link not in self.visited_urls:
                            to_visit.append(link)
                
                # Mark as visited
                self.visited_urls.add(url)
                domain = self._get_domain(url)
                self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
                self.stats["pages_scraped"] += 1
                self.stats["total_text_collected"] += len(text)
                
            except Exception as e:
                self.logger.error(f"Error crawling {url}: {e}")
                self.stats["pages_failed"] += 1
                self.stats["errors"].append({"url": url, "error": str(e)})
        
        self.logger.info(f"Crawling completed. Collected {len(texts)} texts")
        return texts
    
    def get_statistics(self) -> Dict:
        """
        Get scraping statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "pages_scraped": self.stats["pages_scraped"],
            "pages_failed": self.stats["pages_failed"],
            "pages_blocked": self.stats["pages_blocked"],
            "total_text_collected": self.stats["total_text_collected"],
            "unique_domains": len(self.domain_counts),
            "total_errors": len(self.stats["errors"])
        }
    
    def reset_statistics(self):
        """Reset scraping statistics and tracking."""
        self.visited_urls.clear()
        self.domain_counts.clear()
        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "pages_blocked": 0,
            "total_text_collected": 0,
            "errors": []
        }
        self.logger.info("Statistics reset")
