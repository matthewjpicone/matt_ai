"""
Tests for the Instruction Handler module
"""

import unittest
import json
import tempfile
from pathlib import Path
from src.matt_ai.instruction_handler import InstructionHandler


class TestInstructionHandler(unittest.TestCase):
    """Test cases for InstructionHandler class."""
    
    def setUp(self):
        """Set up test fixtures with temporary file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_file.close()
        self.handler = InstructionHandler(instruction_file=self.temp_file.name)
    
    def tearDown(self):
        """Clean up temporary file."""
        temp_path = Path(self.temp_file.name)
        if temp_path.exists():
            temp_path.unlink()
    
    def test_initialization(self):
        """Test handler initialization."""
        self.assertIsNotNone(self.handler)
        self.assertEqual(len(self.handler.instructions), 0)
        self.assertEqual(len(self.handler.active_skills), 0)
    
    def test_add_instruction_success(self):
        """Test adding a valid instruction."""
        result = self.handler.add_instruction(
            instruction="When explaining Python, emphasize its readability and simplicity.",
            skill_name="python_explanation",
            category="technical"
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(len(self.handler.instructions), 1)
        self.assertIn("python_explanation", self.handler.active_skills)
    
    def test_add_instruction_too_short(self):
        """Test adding an instruction that's too short."""
        result = self.handler.add_instruction(
            instruction="Short",
            skill_name="test"
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(len(self.handler.instructions), 0)
    
    def test_add_instruction_without_skill_name(self):
        """Test adding instruction without explicit skill name."""
        result = self.handler.add_instruction(
            instruction="This is a valid instruction for testing purposes.",
            skill_name=None
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(len(self.handler.instructions), 1)
        # Should generate a default skill name
        self.assertTrue(self.handler.instructions[0]["skill_name"].startswith("skill_"))
    
    def test_get_active_instructions(self):
        """Test retrieving active instructions."""
        # Add some instructions
        self.handler.add_instruction(
            "Test instruction 1",
            skill_name="skill1"
        )
        self.handler.add_instruction(
            "Test instruction 2",
            skill_name="skill2"
        )
        
        # Deactivate one
        self.handler.deactivate_instruction(1)
        
        active = self.handler.get_active_instructions()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["skill_name"], "skill2")
    
    def test_get_instructions_by_category(self):
        """Test filtering instructions by category."""
        self.handler.add_instruction(
            "Technical instruction",
            skill_name="tech1",
            category="technical"
        )
        self.handler.add_instruction(
            "Creative instruction",
            skill_name="creative1",
            category="creative"
        )
        
        technical = self.handler.get_instructions_by_category("technical")
        self.assertEqual(len(technical), 1)
        self.assertEqual(technical[0]["category"], "technical")
    
    def test_mark_instruction_applied(self):
        """Test marking instruction as applied."""
        self.handler.add_instruction(
            "Test instruction",
            skill_name="test"
        )
        
        instruction_id = self.handler.instructions[0]["id"]
        self.assertFalse(self.handler.instructions[0]["training_applied"])
        
        self.handler.mark_instruction_applied(instruction_id)
        self.assertTrue(self.handler.instructions[0]["training_applied"])
    
    def test_deactivate_instruction(self):
        """Test deactivating an instruction."""
        self.handler.add_instruction(
            "Test instruction",
            skill_name="test_skill"
        )
        
        instruction_id = self.handler.instructions[0]["id"]
        self.assertEqual(self.handler.instructions[0]["status"], "active")
        self.assertIn("test_skill", self.handler.active_skills)
        
        self.handler.deactivate_instruction(instruction_id)
        self.assertEqual(self.handler.instructions[0]["status"], "inactive")
        self.assertNotIn("test_skill", self.handler.active_skills)
    
    def test_get_training_prompts(self):
        """Test generating training prompts from instructions."""
        self.handler.add_instruction(
            "Explain concepts clearly",
            skill_name="clear_explanation",
            category="general"
        )
        
        prompts = self.handler.get_training_prompts()
        self.assertEqual(len(prompts), 1)
        self.assertIn("clear_explanation", prompts[0])
        self.assertIn("Explain concepts clearly", prompts[0])
    
    def test_format_instruction_for_chat(self):
        """Test formatting instruction for chat."""
        instruction = "Test instruction text"
        formatted = self.handler.format_instruction_for_chat(instruction)
        
        self.assertIn("[User Instruction]", formatted)
        self.assertIn("Test instruction text", formatted)
    
    def test_get_instruction_summary(self):
        """Test getting instruction summary."""
        # Add multiple instructions
        self.handler.add_instruction(
            "Instruction 1",
            skill_name="skill1",
            category="technical"
        )
        self.handler.add_instruction(
            "Instruction 2",
            skill_name="skill2",
            category="creative"
        )
        self.handler.deactivate_instruction(1)
        
        summary = self.handler.get_instruction_summary()
        
        self.assertEqual(summary["total_instructions"], 2)
        self.assertEqual(summary["active_instructions"], 1)
        self.assertEqual(summary["inactive_instructions"], 1)
        self.assertEqual(summary["categories"]["technical"], 1)
        self.assertEqual(summary["categories"]["creative"], 1)
    
    def test_clear_all_instructions(self):
        """Test clearing all instructions."""
        self.handler.add_instruction(
            "Instruction 1",
            skill_name="skill1"
        )
        self.handler.add_instruction(
            "Instruction 2",
            skill_name="skill2"
        )
        
        self.assertEqual(len(self.handler.instructions), 2)
        
        self.handler.clear_all_instructions()
        
        self.assertEqual(len(self.handler.instructions), 0)
        self.assertEqual(len(self.handler.active_skills), 0)
    
    def test_persistence(self):
        """Test that instructions are persisted to file."""
        # Add instruction
        self.handler.add_instruction(
            "Persistent instruction",
            skill_name="persistent",
            category="test"
        )
        
        # Create new handler with same file
        new_handler = InstructionHandler(instruction_file=self.temp_file.name)
        
        # Should load previous instructions
        self.assertEqual(len(new_handler.instructions), 1)
        self.assertEqual(new_handler.instructions[0]["skill_name"], "persistent")
        self.assertIn("persistent", new_handler.active_skills)


if __name__ == '__main__':
    unittest.main()
