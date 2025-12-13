# Matt AI - Self-Training Large Language Model

A self-improving AI system with built-in ethical standards and constraints. This project implements a large language model that can iteratively improve itself through training while adhering to strict ethical guidelines defined in a read-only configuration file.

## Features

- 🧠 **Self-Training LLM**: Transformer-based language model with iterative improvement capabilities
- ⚖️ **Ethical Standards Controller**: Enforces ethical principles and operational constraints
- 🔒 **Read-Only Ethics Config**: Immutable ethical standards defined in `ethical_standards.json`
- 🇬🇧 **English Language Focus**: Optimized for English language processing
- 📈 **Iterative Improvement**: Progressive self-training with quality evaluation
- 🔍 **Continuous Monitoring**: Tracks violations and ensures compliance
- 💾 **Checkpoint System**: Saves model states during training

## Architecture

### Core Components

1. **EthicalController** (`src/matt_ai/ethical_controller.py`)
   - Loads and enforces ethical standards from configuration
   - Validates actions and outputs against defined principles
   - Logs violations and triggers safety measures

2. **SelfTrainingLLM** (`src/matt_ai/model.py`)
   - Transformer-based language model
   - Text generation with ethical validation
   - Training and evaluation capabilities

3. **IterativeTrainer** (`src/matt_ai/trainer.py`)
   - Manages training loops with gradient accumulation
   - Implements self-improvement cycles
   - Evaluates and filters generated content

4. **DataPreparer** (`src/matt_ai/data_utils.py`)
   - Prepares and cleans English text data
   - Handles data augmentation and quality control

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

```bash
# Clone the repository
git clone https://github.com/matthewjpicone/matt_ai.git
cd matt_ai

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The system provides four main operation modes:

### 1. Information Mode

Display system information and ethical standards:

```bash
python main.py info
```

View information about a trained model:

```bash
python main.py info --model-path checkpoints/best_model
```

### 2. Training Mode

Train the model on text data:

```bash
# Train with sample data
python main.py train --epochs 3 --batch-size 4

# Train with custom data file
python main.py train --data-file my_data.txt --epochs 5 --output-dir models/my_model

# Advanced training options
python main.py train \
  --model-name gpt2 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --max-iterations 100 \
  --save-every 10 \
  --eval-every 5 \
  --checkpoint-dir checkpoints \
  --output-dir models/trained_model
```

### 3. Generation Mode

Generate text with the model:

```bash
# Generate with sample prompts
python main.py generate

# Generate from specific prompt
python main.py generate --prompt "Artificial intelligence will"

# Generate from trained model
python main.py generate \
  --model-path checkpoints/best_model \
  --prompt "The future of technology" \
  --max-length 150 \
  --temperature 0.8 \
  --num-sequences 3
```

### 4. Self-Improvement Mode

Run self-improvement cycles:

```bash
# Basic self-improvement
python main.py self-improve --cycles 3

# Advanced self-improvement
python main.py self-improve \
  --model-path checkpoints/my_model \
  --cycles 5 \
  --generations-per-cycle 15 \
  --batch-size 4 \
  --output-dir models/improved_model
```

## Ethical Standards

The system enforces ethical standards defined in `ethical_standards.json`. This file is set to read-only to ensure consistent ethical behavior.

### Core Ethical Principles

1. Do no harm to humans or society
2. Respect human autonomy and dignity
3. Promote fairness and prevent discrimination
4. Protect privacy and data security
5. Operate transparently within defined boundaries

### Prohibited Actions

- Generate content that promotes violence or harm
- Create content that spreads misinformation deliberately
- Generate personally identifiable information without consent
- Produce content that discriminates based on protected characteristics
- Assist in illegal activities or fraud
- Generate content that exploits or harms children
- Create deepfakes or impersonations without disclosure

### Operational Constraints

- Maximum response length: 4096 characters
- Maximum training iterations per session: 100
- Automatic shutdown on ethical violations
- Comprehensive audit logging
- Human review required for high-confidence outputs

### Learning Guidelines

- Prioritize factual accuracy
- Verify sources when possible
- Acknowledge uncertainty
- Avoid reinforcing biases
- Continuously audit outputs

## Configuration

### Ethical Standards (`ethical_standards.json`)

This read-only configuration file controls all ethical aspects of the AI:

```json
{
  "version": "1.0.0",
  "ethical_standards": {
    "core_principles": [...],
    "prohibited_actions": [...],
    "content_filters": {...},
    "operational_constraints": {...},
    "learning_guidelines": {...},
    "safety_measures": {...}
  }
}
```

**Important**: This file is automatically set to read-only mode to prevent unauthorized modifications.

## Development

### Project Structure

```
matt_ai/
├── src/
│   └── matt_ai/
│       ├── __init__.py
│       ├── ethical_controller.py
│       ├── model.py
│       ├── trainer.py
│       └── data_utils.py
├── main.py
├── ethical_standards.json
├── requirements.txt
├── README.md
└── .gitignore
```

### Logging

All operations are logged to:
- Console output (stdout)
- Log file: `matt_ai.log`

Adjust log level with `--log-level` flag:
```bash
python main.py train --log-level DEBUG
```

## Technical Details

### Model Architecture

- Based on GPT-2 transformer architecture
- Configurable layer depth and attention heads
- Support for custom tokenization
- Device-agnostic (CPU/CUDA)

### Training Process

1. **Data Preparation**: Clean and tokenize English text
2. **Ethical Validation**: Check compliance with ethical standards
3. **Iterative Training**: Progressive improvement through multiple epochs
4. **Quality Evaluation**: Assess perplexity and loss metrics
5. **Checkpoint Saving**: Preserve best model states
6. **Self-Improvement**: Generate and learn from high-quality outputs

### Self-Improvement Cycle

1. Generate new text from seed prompts
2. Evaluate quality of generated content
3. Filter for high-quality generations
4. Train on filtered high-quality data
5. Repeat for multiple cycles

## Safety Features

- **Killswitch**: Emergency shutdown capability
- **Rate Limiting**: Prevents resource abuse
- **Audit Logging**: Complete operation history
- **Violation Tracking**: Records ethical breaches
- **Authentication**: Required for sensitive operations
- **Auto-Shutdown**: Triggers on ethical violations

## Limitations

- Requires significant computational resources for large-scale training
- Quality depends on training data
- English language focused (extensible to other languages)
- Self-improvement effectiveness varies with seed data quality
- May require manual review for production deployments

## Future Enhancements

- Multi-language support beyond English
- Advanced reinforcement learning from human feedback (RLHF)
- Distributed training capabilities
- Enhanced quality evaluation metrics
- Web API interface
- Fine-tuning on domain-specific tasks
- Integration with external knowledge bases

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure all contributions:
- Maintain ethical standards compliance
- Include appropriate tests
- Follow existing code style
- Update documentation as needed

## Acknowledgments

Built on top of:
- PyTorch for deep learning
- Hugging Face Transformers for model architecture
- Python ecosystem for data processing

## Contact

Matthew J Picone
GitHub: [@matthewjpicone](https://github.com/matthewjpicone)

---

**Note**: This is a development system. For production deployments, conduct thorough testing, security audits, and human oversight.

