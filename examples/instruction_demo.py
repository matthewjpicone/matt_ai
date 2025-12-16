#!/usr/bin/env python3
"""
Instruction Handler Demo

Demonstrates how to teach the AI new skills programmatically.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matt_ai.instruction_handler import InstructionHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run instruction handler demo."""
    logger.info("=" * 50)
    logger.info("Instruction Handler Demo")
    logger.info("=" * 50)
    
    # Initialize instruction handler
    handler = InstructionHandler(instruction_file="data/demo_instructions.json")
    
    logger.info("\nAdding instructions to teach the AI new skills...\n")
    
    # Add technical instruction
    result1 = handler.add_instruction(
        instruction=(
            "When explaining Python programming, emphasize its readability and simplicity. "
            "Mention that it's widely used for web development, data science, machine learning, "
            "and automation. Highlight features like dynamic typing, extensive libraries, and "
            "a supportive community."
        ),
        skill_name="python_explanation",
        category="technical"
    )
    logger.info(f"Technical skill: {result1['message']}")
    
    # Add creative instruction
    result2 = handler.add_instruction(
        instruction=(
            "When writing creative stories, focus on vivid descriptions and engaging narratives. "
            "Use sensory details to bring scenes to life. Develop characters with depth and "
            "motivations. Build tension and resolution in your plot. Use metaphors and similes "
            "to create powerful imagery."
        ),
        skill_name="creative_storytelling",
        category="creative"
    )
    logger.info(f"Creative skill: {result2['message']}")
    
    # Add educational instruction
    result3 = handler.add_instruction(
        instruction=(
            "When teaching new concepts, break them down into simple, digestible steps. "
            "Start with foundational ideas before moving to complex topics. Use examples "
            "and analogies that relate to everyday experiences. Check for understanding "
            "at each stage. Encourage questions and provide clear explanations."
        ),
        skill_name="step_by_step_teaching",
        category="educational"
    )
    logger.info(f"Educational skill: {result3['message']}")
    
    # Add conversational instruction
    result4 = handler.add_instruction(
        instruction=(
            "In conversations, be friendly, respectful, and engaging. Listen carefully to "
            "what the user is asking. Provide helpful, accurate information. If you don't "
            "know something, be honest about it. Ask clarifying questions when needed. "
            "Maintain a positive and supportive tone."
        ),
        skill_name="conversational_style",
        category="conversational"
    )
    logger.info(f"Conversational skill: {result4['message']}")
    
    # Display summary
    logger.info("\n--- Instruction Summary ---")
    summary = handler.get_instruction_summary()
    logger.info(f"Total instructions: {summary['total_instructions']}")
    logger.info(f"Active instructions: {summary['active_instructions']}")
    logger.info(f"Active skills: {', '.join(summary['active_skills'])}")
    
    logger.info("\nCategories breakdown:")
    for category, count in summary['categories'].items():
        logger.info(f"  - {category}: {count}")
    
    # Get training prompts
    logger.info("\n--- Training Prompts ---")
    prompts = handler.get_training_prompts()
    logger.info(f"Generated {len(prompts)} training prompts from instructions\n")
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Prompt {i}:")
        logger.info(prompt[:200] + "...\n")
    
    # Demonstrate instruction filtering
    logger.info("--- Instructions by Category ---")
    technical_instructions = handler.get_instructions_by_category("technical")
    logger.info(f"Technical instructions: {len(technical_instructions)}")
    for inst in technical_instructions:
        logger.info(f"  - {inst['skill_name']}: {inst['instruction'][:80]}...")
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo completed!")
    logger.info(f"Instructions saved to: data/demo_instructions.json")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
