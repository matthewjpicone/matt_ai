"""
Instruction Handler Module

Allows users to pass instructions to the LLM to teach it new skills.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


class InstructionHandler:
    """
    Handles user instructions to teach the LLM new skills.
    
    Processes and stores instructions that guide the model's behavior
    and learning objectives.
    """
    
    def __init__(self, instruction_file: str = "instructions.json"):
        """
        Initialize the instruction handler.
        
        Args:
            instruction_file: Path to store instruction history
        """
        self.logger = logging.getLogger(__name__)
        self.instruction_file = Path(instruction_file)
        self.instructions: List[Dict] = []
        self.active_skills: Set[str] = set()
        
        # Load existing instructions if available
        self._load_instructions()
    
    def _load_instructions(self):
        """Load instruction history from file."""
        if self.instruction_file.exists():
            try:
                with open(self.instruction_file, 'r') as f:
                    data = json.load(f)
                    self.instructions = data.get("instructions", [])
                    self.active_skills = set(data.get("active_skills", []))
                self.logger.info(f"Loaded {len(self.instructions)} instructions")
            except Exception as e:
                self.logger.error(f"Error loading instructions: {e}")
    
    def _save_instructions(self):
        """Save instruction history to file."""
        try:
            self.instruction_file.parent.mkdir(exist_ok=True, parents=True)
            with open(self.instruction_file, 'w') as f:
                json.dump({
                    "instructions": self.instructions,
                    "active_skills": list(self.active_skills)
                }, f, indent=2)
            self.logger.info("Instructions saved")
        except Exception as e:
            self.logger.error(f"Error saving instructions: {e}")
    
    def add_instruction(
        self,
        instruction: str,
        skill_name: Optional[str] = None,
        category: str = "general"
    ) -> Dict:
        """
        Add a new instruction to teach the LLM.
        
        Args:
            instruction: The instruction text
            skill_name: Optional name for the skill being taught
            category: Category of the instruction (e.g., "general", "technical", "creative")
            
        Returns:
            Dictionary with instruction details and status
        """
        # Validate instruction
        if not instruction or len(instruction.strip()) < 10:
            return {
                "success": False,
                "message": "Instruction too short. Please provide detailed guidance."
            }
        
        # Create instruction record
        instruction_record = {
            "id": len(self.instructions) + 1,
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction.strip(),
            "skill_name": skill_name or f"skill_{len(self.instructions) + 1}",
            "category": category,
            "status": "active",
            "training_applied": False
        }
        
        self.instructions.append(instruction_record)
        
        # Add the skill name (either provided or auto-generated) to active skills
        self.active_skills.add(instruction_record["skill_name"])
        
        self._save_instructions()
        
        self.logger.info(f"Added instruction: {skill_name or 'unnamed'}")
        
        return {
            "success": True,
            "message": f"Instruction added successfully: {skill_name or 'unnamed'}",
            "instruction_id": instruction_record["id"]
        }
    
    def get_active_instructions(self) -> List[Dict]:
        """
        Get all active instructions.
        
        Returns:
            List of active instruction records
        """
        return [inst for inst in self.instructions if inst["status"] == "active"]
    
    def get_instructions_by_category(self, category: str) -> List[Dict]:
        """
        Get instructions by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of instructions in the category
        """
        return [inst for inst in self.instructions if inst["category"] == category]
    
    def mark_instruction_applied(self, instruction_id: int):
        """
        Mark an instruction as applied to training.
        
        Args:
            instruction_id: ID of the instruction
        """
        for inst in self.instructions:
            if inst["id"] == instruction_id:
                inst["training_applied"] = True
                self._save_instructions()
                self.logger.info(f"Marked instruction {instruction_id} as applied")
                break
    
    def deactivate_instruction(self, instruction_id: int):
        """
        Deactivate an instruction.
        
        Args:
            instruction_id: ID of the instruction to deactivate
        """
        for inst in self.instructions:
            if inst["id"] == instruction_id:
                inst["status"] = "inactive"
                if inst["skill_name"] in self.active_skills:
                    self.active_skills.remove(inst["skill_name"])
                self._save_instructions()
                self.logger.info(f"Deactivated instruction {instruction_id}")
                break
    
    def get_training_prompts(self) -> List[str]:
        """
        Generate training prompts from active instructions.
        
        Returns:
            List of prompts for training the model
        """
        prompts = []
        
        for inst in self.get_active_instructions():
            # Convert instruction to training prompt
            prompt = f"Task: {inst['skill_name']}\n"
            prompt += f"Instructions: {inst['instruction']}\n"
            prompt += f"Category: {inst['category']}\n\n"
            prompt += "Please demonstrate understanding of this task."
            
            prompts.append(prompt)
        
        return prompts
    
    def format_instruction_for_chat(self, instruction: str) -> str:
        """
        Format an instruction for inclusion in chat context.
        
        Args:
            instruction: The instruction text
            
        Returns:
            Formatted instruction
        """
        return f"[User Instruction]: {instruction}"
    
    def get_instruction_summary(self) -> Dict:
        """
        Get summary of all instructions.
        
        Returns:
            Dictionary with instruction statistics and summary
        """
        active = self.get_active_instructions()
        
        categories = {}
        for inst in self.instructions:
            cat = inst["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_instructions": len(self.instructions),
            "active_instructions": len(active),
            "inactive_instructions": len(self.instructions) - len(active),
            "active_skills": list(self.active_skills),
            "categories": categories,
            "recent_instructions": self.instructions[-5:] if self.instructions else []
        }
    
    def clear_all_instructions(self):
        """Clear all instructions (use with caution)."""
        self.instructions = []
        self.active_skills = set()
        self._save_instructions()
        self.logger.warning("All instructions cleared")
