"""
Tests for the Web Scraper module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.matt_ai.web_scraper import WebScraper


class TestWebScraper(unittest.TestCase):
    """Test cases for WebScraper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = WebScraper(
            rate_limit_delay=0.1,  # Faster for testing
            max_pages_per_domain=5
        )
    
    def test_initialization(self):
        """Test scraper initialization."""
        self.assertEqual(self.scraper.rate_limit_delay, 0.1)
        self.assertEqual(self.scraper.max_pages_per_domain, 5)
        self.assertEqual(len(self.scraper.visited_urls), 0)
        self.assertEqual(self.scraper.stats["pages_scraped"], 0)
    
    def test_get_domain(self):
        """Test domain extraction from URL."""
        url = "https://example.com/path/to/page"
        domain = self.scraper._get_domain(url)
        self.assertEqual(domain, "example.com")
    
    def test_should_scrape_domain_limit(self):
        """Test domain limit checking."""
        url = "https://example.com/page1"
        
        # Should be able to scrape initially
        self.assertTrue(self.scraper._should_scrape_domain(url))
        
        # Simulate reaching domain limit
        self.scraper.domain_counts["example.com"] = 5
        self.assertFalse(self.scraper._should_scrape_domain(url))
    
    def test_extract_text(self):
        """Test text extraction from HTML."""
        from bs4 import BeautifulSoup
        
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <nav>Navigation</nav>
                <h1>Hello World</h1>
                <p>This is a test paragraph.</p>
                <script>console.log('test');</script>
                <footer>Footer</footer>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        text = self.scraper._extract_text(soup)
        
        # Should contain main content but not script/nav/footer
        self.assertIn("Hello World", text)
        self.assertIn("test paragraph", text)
        self.assertNotIn("console.log", text)
    
    def test_extract_links(self):
        """Test link extraction from HTML."""
        from bs4 import BeautifulSoup
        
        html = """
        <html>
            <body>
                <a href="https://example.com/page1">Link 1</a>
                <a href="/relative/path">Link 2</a>
                <a href="#fragment">Link 3</a>
                <a href="mailto:test@example.com">Email</a>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        base_url = "https://example.com"
        links = self.scraper._extract_links(soup, base_url)
        
        # Should extract and resolve links, but not fragments or non-HTTP
        self.assertIn("https://example.com/page1", links)
        self.assertIn("https://example.com/relative/path", links)
        # Should not include fragments or mailto
        self.assertNotIn("mailto:test@example.com", [l for l in links if 'mailto' in l])
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Add some test data
        self.scraper.stats["pages_scraped"] = 10
        self.scraper.stats["pages_failed"] = 2
        self.scraper.domain_counts["example.com"] = 5
        
        stats = self.scraper.get_statistics()
        
        self.assertEqual(stats["pages_scraped"], 10)
        self.assertEqual(stats["pages_failed"], 2)
        self.assertEqual(stats["unique_domains"], 1)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Add some test data
        self.scraper.visited_urls.add("https://example.com")
        self.scraper.stats["pages_scraped"] = 10
        self.scraper.domain_counts["example.com"] = 5
        
        # Reset
        self.scraper.reset_statistics()
        
        # Verify everything is cleared
        self.assertEqual(len(self.scraper.visited_urls), 0)
        self.assertEqual(self.scraper.stats["pages_scraped"], 0)
        self.assertEqual(len(self.scraper.domain_counts), 0)
    
    @patch('src.matt_ai.web_scraper.requests.Session.get')
    @patch('time.sleep')
    def test_scrape_url_success(self, mock_sleep, mock_get):
        """Test successful URL scraping."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b"<html><body><h1>Test Content</h1></body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock robots.txt to allow
        with patch.object(self.scraper, '_can_fetch', return_value=True):
            text = self.scraper.scrape_url("https://example.com/page1")
        
        self.assertIsNotNone(text)
        self.assertIn("Test Content", text)
        self.assertEqual(self.scraper.stats["pages_scraped"], 1)
    
    @patch('src.matt_ai.web_scraper.requests.Session.get')
    def test_scrape_url_failure(self, mock_get):
        """Test URL scraping failure."""
        # Mock request exception
        mock_get.side_effect = Exception("Connection error")
        
        with patch.object(self.scraper, '_can_fetch', return_value=True):
            with patch('time.sleep'):
                text = self.scraper.scrape_url("https://example.com/page1")
        
        self.assertIsNone(text)
        self.assertEqual(self.scraper.stats["pages_failed"], 1)
    
    def test_scrape_url_already_visited(self):
        """Test that already visited URLs are skipped."""
        url = "https://example.com/page1"
        self.scraper.visited_urls.add(url)
        
        text = self.scraper.scrape_url(url)
        
        self.assertIsNone(text)
        self.assertEqual(self.scraper.stats["pages_scraped"], 0)


if __name__ == '__main__':
    unittest.main()
