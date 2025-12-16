"""
Tests for Ethical Controller

These tests verify the ethical standards enforcement without requiring heavy ML dependencies.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matt_ai.ethical_controller import EthicalController


class TestEthicalController(unittest.TestCase):
    """Test cases for EthicalController."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = EthicalController()
    
    def test_load_standards(self):
        """Test that ethical standards are loaded correctly."""
        self.assertIsNotNone(self.controller.standards)
        self.assertIn("ethical_standards", self.controller.standards)
        self.assertIn("version", self.controller.standards)
    
    def test_core_principles(self):
        """Test that core principles are accessible."""
        principles = self.controller.get_core_principles()
        self.assertIsInstance(principles, list)
        self.assertGreater(len(principles), 0)
        
        # Check that principles are non-empty strings
        for principle in principles:
            self.assertIsInstance(principle, str)
            self.assertGreater(len(principle), 0)
    
    def test_learning_guidelines(self):
        """Test that learning guidelines are accessible."""
        guidelines = self.controller.get_learning_guidelines()
        self.assertIsInstance(guidelines, dict)
        self.assertGreater(len(guidelines), 0)
    
    def test_operational_constraints(self):
        """Test that operational constraints are accessible."""
        constraints = self.controller.get_operational_constraints()
        self.assertIsInstance(constraints, dict)
        self.assertIn("max_response_length", constraints)
        self.assertIn("max_training_iterations_per_session", constraints)
    
    def test_validate_action_positive(self):
        """Test validation of acceptable actions."""
        is_valid, reason = self.controller.validate_action(
            "Generate educational content about science"
        )
        self.assertTrue(is_valid, f"Expected valid action but got: {reason}")
        self.assertEqual(reason, "Action approved")
    
    def test_validate_action_negative(self):
        """Test validation of prohibited actions."""
        # Test violence-related action
        is_valid, reason = self.controller.validate_action(
            "Generate content that promotes violence and harm"
        )
        self.assertFalse(is_valid)
        self.assertIn("violat", reason.lower())
    
    def test_validate_output_length(self):
        """Test output length validation."""
        constraints = self.controller.get_operational_constraints()
        max_length = constraints.get("max_response_length", 4096)
        
        # Valid length
        short_output = "This is a short output"
        is_valid, reason = self.controller.validate_output(short_output)
        self.assertTrue(is_valid)
        
        # Excessive length
        long_output = "x" * (max_length + 1)
        is_valid, reason = self.controller.validate_output(long_output)
        self.assertFalse(is_valid)
        self.assertIn("exceeds", reason.lower())
    
    def test_validate_output_confidence(self):
        """Test output validation with confidence scores."""
        # Low confidence - should pass
        is_valid, reason = self.controller.validate_output("Test output", confidence=0.5)
        self.assertTrue(is_valid)
        
        # High confidence - should suggest review
        is_valid, reason = self.controller.validate_output("Test output", confidence=0.99)
        self.assertTrue(is_valid)
        self.assertIn("review", reason.lower())
    
    def test_training_compliance(self):
        """Test training compliance checks."""
        from datetime import datetime
        
        session_start = datetime.now()
        
        # Within limits
        is_compliant = self.controller.check_training_compliance(10, session_start)
        self.assertTrue(is_compliant)
        
        # Exceeding limits
        is_compliant = self.controller.check_training_compliance(1000, session_start)
        self.assertFalse(is_compliant)
    
    def test_violation_log(self):
        """Test that violations are logged."""
        initial_count = len(self.controller.get_violation_log())
        
        # Trigger a violation
        self.controller.validate_action("Generate content that promotes violence")
        
        final_count = len(self.controller.get_violation_log())
        self.assertGreater(final_count, initial_count)
        
        # Check violation structure
        violations = self.controller.get_violation_log()
        if violations:
            violation = violations[-1]
            self.assertIn("timestamp", violation)
            self.assertIn("action", violation)
            self.assertIn("reason", violation)
    
    def test_config_file_structure(self):
        """Test that the configuration file has proper structure."""
        standards = self.controller.standards
        
        required_sections = [
            "core_principles",
            "prohibited_actions",
            "content_filters",
            "operational_constraints",
            "learning_guidelines",
            "safety_measures"
        ]
        
        ethical_standards = standards.get("ethical_standards", {})
        for section in required_sections:
            self.assertIn(section, ethical_standards, 
                         f"Missing required section: {section}")


class TestEthicalStandardsFile(unittest.TestCase):
    """Test the ethical standards JSON file directly."""
    
    def test_file_exists(self):
        """Test that the ethical standards file exists."""
        config_path = Path(__file__).parent.parent / "ethical_standards.json"
        self.assertTrue(config_path.exists())
    
    def test_valid_json(self):
        """Test that the file contains valid JSON."""
        config_path = Path(__file__).parent.parent / "ethical_standards.json"
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, dict)
    
    def test_version_present(self):
        """Test that version information is present."""
        config_path = Path(__file__).parent.parent / "ethical_standards.json"
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn("version", data)
        self.assertIn("last_updated", data)


if __name__ == '__main__':
    unittest.main()
