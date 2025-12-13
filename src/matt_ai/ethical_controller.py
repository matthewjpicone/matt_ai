"""
Ethical Standards Controller

This module enforces ethical standards and operational constraints
defined in the ethical_standards.json configuration file.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonschema


class EthicalController:
    """
    Enforces ethical standards and operational constraints for the AI system.
    
    This controller loads ethical standards from a read-only configuration file
    and ensures all model operations comply with defined principles and constraints.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Ethical Controller.
        
        Args:
            config_path: Path to ethical standards configuration file.
                        Defaults to ethical_standards.json in project root.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "ethical_standards.json"
        
        self.config_path = Path(config_path)
        self.standards = self._load_standards()
        self.violation_log = []
        
        # Make config file read-only on Unix systems
        if os.name != 'nt':
            try:
                os.chmod(self.config_path, 0o444)
                self.logger.info(f"Ethical standards file set to read-only: {self.config_path}")
            except Exception as e:
                self.logger.warning(f"Could not set read-only permissions: {e}")
    
    def _load_standards(self) -> Dict:
        """Load and validate ethical standards from configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                standards = json.load(f)
            
            self.logger.info(f"Loaded ethical standards v{standards.get('version', 'unknown')}")
            return standards
        except FileNotFoundError:
            self.logger.error(f"Ethical standards file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in ethical standards file: {e}")
            raise
    
    def validate_action(self, action: str, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Validate if an action is ethically permissible.
        
        Args:
            action: Description of the action to validate
            context: Additional context for validation
            
        Returns:
            Tuple of (is_valid, reason)
        """
        context = context or {}
        
        # Check against prohibited actions
        prohibited = self.standards.get("ethical_standards", {}).get("prohibited_actions", [])
        for prohibited_action in prohibited:
            if self._action_matches_prohibition(action, prohibited_action):
                reason = f"Action violates prohibition: {prohibited_action}"
                self._log_violation(action, reason, context)
                return False, reason
        
        return True, "Action approved"
    
    def _action_matches_prohibition(self, action: str, prohibition: str) -> bool:
        """Check if an action matches a prohibited action pattern."""
        # Simple keyword matching - can be enhanced with NLP
        action_lower = action.lower()
        prohibition_lower = prohibition.lower()
        
        # Define key prohibited terms that indicate violations
        prohibited_keywords = [
            'violence', 'harm', 'misinformation', 'discriminat', 
            'illegal', 'fraud', 'exploit', 'deepfake', 'impersonat'
        ]
        
        # Check if action contains prohibited keywords and is related to generation/creation
        contains_prohibited = any(keyword in action_lower for keyword in prohibited_keywords)
        is_generation_action = any(word in action_lower for word in ['generate', 'create', 'produce', 'make'])
        
        # If it's a generation action with prohibited content, flag it
        if contains_prohibited and is_generation_action:
            return True
        
        return False
    
    def validate_output(self, output: str, confidence: float = 0.0) -> Tuple[bool, str]:
        """
        Validate model output against ethical standards.
        
        Args:
            output: The generated text output
            confidence: Confidence score of the generation
            
        Returns:
            Tuple of (is_valid, reason)
        """
        constraints = self.standards.get("ethical_standards", {}).get("operational_constraints", {})
        
        # Check length constraint
        max_length = constraints.get("max_response_length", 4096)
        if len(output) > max_length:
            return False, f"Output exceeds maximum length of {max_length} characters"
        
        # Check if human review is required for high confidence
        review_threshold = constraints.get("require_human_review_above_confidence", 0.95)
        if confidence > review_threshold:
            self.logger.warning(f"Output confidence {confidence} exceeds review threshold")
            return True, f"Human review recommended (confidence: {confidence})"
        
        return True, "Output validated"
    
    def _log_violation(self, action: str, reason: str, context: Dict):
        """Log an ethical violation."""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reason": reason,
            "context": context
        }
        self.violation_log.append(violation)
        self.logger.warning(f"Ethical violation: {reason}")
        
        # Auto-shutdown if configured
        constraints = self.standards.get("ethical_standards", {}).get("operational_constraints", {})
        if constraints.get("auto_shutdown_on_ethical_violation", False):
            self.logger.critical("Auto-shutdown triggered due to ethical violation")
            # In production, this would trigger actual shutdown
    
    def get_core_principles(self) -> List[str]:
        """Get the core ethical principles."""
        return self.standards.get("ethical_standards", {}).get("core_principles", [])
    
    def get_learning_guidelines(self) -> Dict:
        """Get guidelines for learning and improvement."""
        return self.standards.get("ethical_standards", {}).get("learning_guidelines", {})
    
    def get_operational_constraints(self) -> Dict:
        """Get operational constraints."""
        return self.standards.get("ethical_standards", {}).get("operational_constraints", {})
    
    def check_training_compliance(self, iteration: int, session_start: datetime) -> bool:
        """
        Check if training iteration is within ethical constraints.
        
        Args:
            iteration: Current training iteration number
            session_start: When the training session started
            
        Returns:
            True if compliant, False otherwise
        """
        constraints = self.get_operational_constraints()
        max_iterations = constraints.get("max_training_iterations_per_session", 100)
        
        if iteration > max_iterations:
            self.logger.warning(f"Training iteration {iteration} exceeds maximum {max_iterations}")
            return False
        
        return True
    
    def get_violation_log(self) -> List[Dict]:
        """Get the log of ethical violations."""
        return self.violation_log.copy()
