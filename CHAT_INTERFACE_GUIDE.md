# Chat Interface Guide

This guide explains how to use Matt AI's interactive chat interface with web scraping and autonomous learning capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Features Overview](#features-overview)
3. [Using the Chat Interface](#using-the-chat-interface)
4. [Teaching New Skills](#teaching-new-skills)
5. [Web Scraping](#web-scraping)
6. [Training the Model](#training-the-model)
7. [Monitoring Status](#monitoring-status)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

First, ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### Launch the Interface

Basic launch:

```bash
python chat_app.py
```

The interface will start on `http://localhost:7860`. Open this URL in your web browser.

### With Custom Settings

```bash
python chat_app.py \
  --model-name gpt2 \
  --rate-limit 2.0 \
  --max-pages-per-domain 10 \
  --batch-size 4 \
  --learning-rate 5e-5 \
  --server-port 7860
```

### Public Access

To create a public shareable link (useful for demos or remote access):

```bash
python chat_app.py --share
```

## Features Overview

### 💬 Real-Time Chat

- Interactive conversation with the AI model
- Responses are validated against ethical standards
- Chat history is maintained during the session

### 📚 Instruction System

- Teach the AI new skills by providing instructions
- Instructions are stored and applied to training
- Categorize instructions (technical, creative, educational, etc.)
- View and manage active skills

### 🌐 Web Scraping

- Autonomous data collection from websites
- **Ethical scraping:**
  - Respects robots.txt files
  - Implements rate limiting (default: 2 seconds between requests)
  - Honors domain limits (default: 10 pages per domain)
- Real-time progress monitoring

### 🎓 Live Training

- Train the model on scraped data in the background
- Continue chatting while training progresses
- Automatic integration of user instructions into training

### 📊 Status Monitoring

- View system statistics
- Track scraping progress
- Monitor training status
- See model information

## Using the Chat Interface

### Basic Conversation

1. Go to the **Chat** tab
2. Type your message in the text box
3. Click "Send" or press Enter
4. The AI will respond based on its current training

### Example Conversations

```
You: What is artificial intelligence?
AI: Artificial intelligence refers to the simulation of human intelligence 
in machines that are programmed to think and learn...

You: Can you explain machine learning?
AI: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed...
```

### Tips for Better Conversations

- Be clear and specific in your questions
- Provide context when needed
- The AI's knowledge is based on its training data
- Responses are ethically validated

## Teaching New Skills

### Adding Instructions

1. Go to the **Teach Skills** tab
2. Fill in the instruction details:
   - **Instruction**: Detailed description of what you want to teach
   - **Skill Name**: A memorable name for this skill
   - **Category**: Choose from general, technical, creative, educational, or conversational

### Example Instructions

#### Technical Skill

```
Instruction: When explaining Python programming, emphasize its readability, 
simplicity, and versatility. Mention popular use cases like web development, 
data science, and automation.

Skill Name: python_explanation
Category: technical
```

#### Creative Skill

```
Instruction: When writing creative content, focus on vivid descriptions, 
engaging narratives, and emotional resonance. Use metaphors and sensory details.

Skill Name: creative_writing
Category: creative
```

#### Educational Skill

```
Instruction: When teaching concepts, break them down into simple steps. 
Start with the basics, provide examples, and build up to more complex ideas.

Skill Name: step_by_step_teaching
Category: educational
```

### Viewing Active Instructions

Click "Refresh Summary" to see:
- Total number of instructions
- Active instructions and skills
- Instructions by category
- Recent additions

## Web Scraping

### Starting a Scrape

1. Go to the **Web Scraping** tab
2. Enter URLs (one per line) in the text box
3. Set the maximum number of pages to scrape
4. Click "Start Scraping"

### Example URLs

```
https://en.wikipedia.org/wiki/Artificial_intelligence
https://en.wikipedia.org/wiki/Machine_learning
https://en.wikipedia.org/wiki/Deep_learning
```

### Ethical Considerations

The scraper automatically:
- Checks robots.txt before scraping
- Waits between requests (rate limiting)
- Limits pages per domain
- Handles errors gracefully
- Logs all activities

### Scraping Status

The scraper provides real-time updates:
- "Scraping web pages..." - In progress
- "Scraping completed. Collected X texts" - Success
- Error messages if issues occur

### Clearing Scraped Data

Click "Clear Scraped Data" to remove all collected texts and reset statistics.

## Training the Model

### Starting Training

1. First, scrape some URLs to collect training data
2. Go to the **Training** tab
3. Select the number of epochs (recommended: 2-3)
4. Click "Start Training"

### Training Process

The training happens in the background:
- You can continue chatting while training runs
- Training status updates automatically
- Instructions are incorporated into training data
- Model checkpoints are saved periodically

### Training with Instructions

When you have active instructions:
1. Add your instructions in the "Teach Skills" tab
2. Scrape relevant data
3. Start training
4. The model will learn from both scraped data and your instructions

### Example Workflow

```
1. Add instruction about Python programming
2. Scrape Python documentation and tutorials
3. Start training for 2-3 epochs
4. Chat with the AI to test its Python knowledge
5. Add more instructions or scrape more data as needed
6. Repeat training to refine the model
```

## Monitoring Status

### System Information

Go to the **Status** tab to view:

- **Model Information:**
  - Model name and architecture
  - Number of parameters
  - Device (CPU/CUDA)

- **Activity Status:**
  - Current training status
  - Scraping status
  - Active/idle state

- **Data Statistics:**
  - Number of scraped texts
  - Pages scraped/failed/blocked
  - Unique domains visited

### Auto-Refresh

The status tab automatically refreshes every 5 seconds to show current information.

### Manual Refresh

Click "Refresh Status" for immediate update.

## Best Practices

### For Web Scraping

1. **Start Small**: Begin with 5-10 pages to test
2. **Respect Limits**: Don't scrape too aggressively
3. **Choose Quality**: Select authoritative, well-written sources
4. **Verify Content**: Check that scraped data is relevant
5. **Be Patient**: Rate limiting means scraping takes time

### For Teaching Skills

1. **Be Specific**: Provide detailed, clear instructions
2. **Use Examples**: Include examples in your instructions
3. **Categorize**: Properly categorize for better organization
4. **Iterate**: Refine instructions based on results
5. **Test**: Chat with the AI to verify learning

### For Training

1. **Quality Over Quantity**: Better to train on less high-quality data
2. **Multiple Epochs**: Use 2-3 epochs for better learning
3. **Monitor**: Watch for improvements in responses
4. **Save Checkpoints**: Training automatically saves progress
5. **Combine Sources**: Mix scraped data with instructions

### For Chat

1. **Clear Questions**: Ask specific, well-formed questions
2. **Provide Context**: Give background when needed
3. **Be Patient**: Initial responses may be basic
4. **Test Learning**: Ask about taught skills to verify training
5. **Iterate**: Refine instructions based on responses

## Troubleshooting

### Interface Won't Start

**Problem**: Error starting the application

**Solutions**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 7860 is available: try `--server-port 7861`
- Review logs in `logs/chat_app.log`

### Scraping Fails

**Problem**: URLs fail to scrape

**Solutions**:
- Check URL validity (must start with http:// or https://)
- Verify internet connectivity
- Website may block scrapers - try different sources
- Check robots.txt compliance - site may disallow scraping
- Reduce rate if getting timeout errors

### Training Doesn't Start

**Problem**: "No scraped data available" error

**Solutions**:
- Scrape some URLs first before training
- Verify scraped texts appear in status tab
- Check that scraping completed successfully
- Try clearing data and rescraping

### Poor AI Responses

**Problem**: AI gives generic or incorrect responses

**Solutions**:
- Add more specific instructions
- Scrape higher-quality data sources
- Train for more epochs (but not too many)
- Test with different prompts
- Verify instructions are active

### Out of Memory

**Problem**: System runs out of memory during training

**Solutions**:
- Reduce batch size: `--batch-size 2`
- Use smaller model: `--model-name distilgpt2`
- Train on less data at a time
- Close other applications
- Consider using a GPU if available

### Slow Performance

**Problem**: Interface or training is slow

**Solutions**:
- Use CPU if CUDA causes issues: `--device cpu`
- Reduce batch size for training
- Scrape fewer pages at a time
- Close unnecessary browser tabs
- Check system resources

## Advanced Usage

### Command-Line Options

```bash
python chat_app.py --help
```

Available options:
- `--model-name`: Base model (default: gpt2)
- `--model-path`: Load pre-trained model
- `--device`: cpu or cuda
- `--rate-limit`: Scraping delay in seconds
- `--max-pages-per-domain`: Domain page limit
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate for training
- `--share`: Create public link
- `--server-port`: Web server port

### Saving and Loading Models

Models are automatically saved during training to `checkpoints/`.

To load a previously trained model:

```bash
python chat_app.py --model-path checkpoints/best_model
```

### Integration with CLI

You can still use the CLI tools alongside the chat interface:

```bash
# Train on custom data
python main.py train --data-file my_data.txt

# Generate text
python main.py generate --prompt "Your prompt"

# View system info
python main.py info
```

## Safety and Ethics

### Ethical Standards

All operations are governed by ethical standards:
- No harmful content generation
- Privacy protection
- Factual accuracy prioritization
- Transparent operation
- Human oversight

### Content Filtering

- Inputs and outputs are validated
- Prohibited content is blocked
- Violations are logged
- Automatic safety measures activate when needed

### Data Privacy

- Scraped data stays local
- No data sent to external servers (unless using --share)
- Chat history is session-only
- Instruction data stored locally

## Support and Contribution

### Getting Help

- Check this guide first
- Review logs in `logs/chat_app.log`
- Check GitHub issues
- Read the main README.md

### Reporting Issues

When reporting issues, include:
- Error messages
- Steps to reproduce
- System information
- Log excerpts

### Contributing

Contributions welcome! Please:
- Follow existing code style
- Add tests for new features
- Update documentation
- Ensure ethical compliance

---

**Happy Learning! 🤖📚**

For more information, see the main [README.md](README.md) file.
