"""
Chat Interface Module

Provides a Gradio-based chat interface for interacting with the LLM
while it trains on scraped data.
"""

import logging
import threading
import time
from typing import List, Tuple, Optional, Dict
import gradio as gr


class ChatInterface:
    """
    Gradio-based chat interface for real-time interaction with the LLM.
    
    Allows users to chat with the model, view training status, and pass instructions.
    """
    
    def __init__(
        self,
        model,
        web_scraper=None,
        instruction_handler=None,
        trainer=None
    ):
        """
        Initialize the chat interface.
        
        Args:
            model: SelfTrainingLLM instance
            web_scraper: WebScraper instance (optional)
            instruction_handler: InstructionHandler instance (optional)
            trainer: IterativeTrainer instance (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.web_scraper = web_scraper
        self.instruction_handler = instruction_handler
        self.trainer = trainer
        
        # Chat history
        self.chat_history: List[Tuple[str, str]] = []
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.training_status = "Idle"
        
        # Scraping state
        self.is_scraping = False
        self.scraping_thread = None
        self.scraped_texts: List[str] = []
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle chat message.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not message or not message.strip():
            return "", history
        
        self.logger.info(f"User: {message}")
        
        try:
            # Generate response
            responses = self.model.generate(
                message,
                max_length=200,
                temperature=0.7,
                num_return_sequences=1
            )
            
            response = responses[0] if responses else "I apologize, I couldn't generate a response."
            
            # Update history
            history.append((message, response))
            
            self.logger.info(f"Assistant: {response}")
            
            return "", history
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def add_instruction(
        self,
        instruction: str,
        skill_name: str,
        category: str
    ) -> str:
        """
        Add a new instruction to teach the LLM.
        
        Args:
            instruction: The instruction text
            skill_name: Name for the skill
            category: Category of the instruction
            
        Returns:
            Status message
        """
        if not self.instruction_handler:
            return "❌ Instruction handler not available"
        
        if not instruction or not instruction.strip():
            return "❌ Please provide an instruction"
        
        result = self.instruction_handler.add_instruction(
            instruction=instruction,
            skill_name=skill_name or None,
            category=category or "general"
        )
        
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
    
    def get_instruction_summary(self) -> str:
        """
        Get summary of instructions.
        
        Returns:
            Formatted summary string
        """
        if not self.instruction_handler:
            return "Instruction handler not available"
        
        summary = self.instruction_handler.get_instruction_summary()
        
        output = "📚 **Instruction Summary**\n\n"
        output += f"**Total Instructions:** {summary['total_instructions']}\n"
        output += f"**Active Instructions:** {summary['active_instructions']}\n"
        output += f"**Active Skills:** {', '.join(summary['active_skills']) if summary['active_skills'] else 'None'}\n\n"
        
        if summary['categories']:
            output += "**Categories:**\n"
            for cat, count in summary['categories'].items():
                output += f"  - {cat}: {count}\n"
        
        if summary['recent_instructions']:
            output += "\n**Recent Instructions:**\n"
            for inst in summary['recent_instructions']:
                output += f"  - {inst['skill_name']}: {inst['instruction'][:50]}...\n"
        
        return output
    
    def start_scraping(self, urls_text: str, max_pages: int) -> str:
        """
        Start web scraping in background.
        
        Args:
            urls_text: URLs to scrape (one per line)
            max_pages: Maximum pages to scrape
            
        Returns:
            Status message
        """
        if not self.web_scraper:
            return "❌ Web scraper not available"
        
        if self.is_scraping:
            return "⚠️ Scraping already in progress"
        
        # Parse URLs
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if not urls:
            return "❌ Please provide at least one URL"
        
        # Start scraping in background
        def scrape_task():
            self.is_scraping = True
            self.training_status = "Scraping web pages..."
            
            try:
                texts = self.web_scraper.scrape_urls(urls, max_pages=max_pages)
                self.scraped_texts.extend(texts)
                self.training_status = f"Scraping completed. Collected {len(texts)} texts."
                self.logger.info(f"Scraping completed: {len(texts)} texts")
            except Exception as e:
                self.training_status = f"Scraping failed: {str(e)}"
                self.logger.error(f"Scraping error: {e}")
            finally:
                self.is_scraping = False
        
        self.scraping_thread = threading.Thread(target=scrape_task, daemon=True)
        self.scraping_thread.start()
        
        return f"✅ Started scraping {len(urls)} URL(s)"
    
    def start_training(self, num_epochs: int) -> str:
        """
        Start training on scraped data.
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            Status message
        """
        if not self.trainer:
            return "❌ Trainer not available"
        
        if self.is_training:
            return "⚠️ Training already in progress"
        
        if not self.scraped_texts:
            return "❌ No scraped data available. Please scrape some URLs first."
        
        # Start training in background
        def training_task():
            self.is_training = True
            self.training_status = f"Training on {len(self.scraped_texts)} texts..."
            
            try:
                from ..matt_ai.data_utils import DataPreparer
                data_preparer = DataPreparer()
                
                # Clean and prepare data
                train_texts, val_texts = data_preparer.prepare_training_data(self.scraped_texts)
                
                # Add instruction-based prompts if available
                if self.instruction_handler:
                    instruction_prompts = self.instruction_handler.get_training_prompts()
                    train_texts.extend(instruction_prompts)
                    self.logger.info(f"Added {len(instruction_prompts)} instruction prompts")
                
                # Train
                results = self.trainer.train(
                    training_texts=train_texts,
                    validation_texts=val_texts,
                    num_epochs=num_epochs,
                    save_every=50,
                    evaluate_every=25
                )
                
                self.training_status = f"Training completed! Loss: {results.get('final_loss', 'N/A')}"
                self.logger.info(f"Training completed: {results}")
                
            except Exception as e:
                self.training_status = f"Training failed: {str(e)}"
                self.logger.error(f"Training error: {e}")
            finally:
                self.is_training = False
        
        self.training_thread = threading.Thread(target=training_task, daemon=True)
        self.training_thread.start()
        
        return f"✅ Started training on {len(self.scraped_texts)} texts for {num_epochs} epoch(s)"
    
    def get_status(self) -> str:
        """
        Get current status.
        
        Returns:
            Status information
        """
        status = "📊 **System Status**\n\n"
        
        # Model info
        model_info = self.model.get_model_info()
        status += f"**Model:** {model_info['model_name']}\n"
        status += f"**Device:** {model_info['device']}\n"
        status += f"**Parameters:** {model_info['parameters']:,}\n\n"
        
        # Training status
        status += f"**Training Status:** {self.training_status}\n"
        status += f"**Is Training:** {'Yes' if self.is_training else 'No'}\n"
        status += f"**Is Scraping:** {'Yes' if self.is_scraping else 'No'}\n\n"
        
        # Data stats
        status += f"**Scraped Texts:** {len(self.scraped_texts)}\n"
        
        if self.web_scraper:
            scraper_stats = self.web_scraper.get_statistics()
            status += f"**Pages Scraped:** {scraper_stats['pages_scraped']}\n"
            status += f"**Pages Failed:** {scraper_stats['pages_failed']}\n"
            status += f"**Pages Blocked:** {scraper_stats['pages_blocked']}\n"
        
        return status
    
    def clear_scraped_data(self) -> str:
        """
        Clear scraped data.
        
        Returns:
            Status message
        """
        count = len(self.scraped_texts)
        self.scraped_texts = []
        
        if self.web_scraper:
            self.web_scraper.reset_statistics()
        
        return f"✅ Cleared {count} scraped texts"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Matt AI - Chat & Training Interface") as interface:
            gr.Markdown("# 🤖 Matt AI - Interactive Learning System")
            gr.Markdown("Chat with the AI, teach it new skills, and watch it learn from the web!")
            
            with gr.Tabs():
                # Chat Tab
                with gr.Tab("💬 Chat"):
                    chatbot = gr.Chatbot(label="Chat with Matt AI", height=400)
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")
                    
                    send_btn.click(
                        self.chat,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                    msg.submit(
                        self.chat,
                        inputs=[msg, chatbot],
                        outputs=[msg, chatbot]
                    )
                    clear_btn.click(lambda: [], outputs=chatbot)
                
                # Instructions Tab
                with gr.Tab("📚 Teach Skills"):
                    gr.Markdown("### Add instructions to teach the AI new skills")
                    
                    instruction_text = gr.Textbox(
                        label="Instruction",
                        placeholder="Example: When asked about Python, explain it as a high-level programming language...",
                        lines=4
                    )
                    skill_name_input = gr.Textbox(
                        label="Skill Name",
                        placeholder="Example: python_explanation"
                    )
                    category_input = gr.Dropdown(
                        label="Category",
                        choices=["general", "technical", "creative", "educational", "conversational"],
                        value="general"
                    )
                    
                    add_inst_btn = gr.Button("Add Instruction", variant="primary")
                    inst_output = gr.Textbox(label="Status", lines=2)
                    
                    add_inst_btn.click(
                        self.add_instruction,
                        inputs=[instruction_text, skill_name_input, category_input],
                        outputs=inst_output
                    )
                    
                    gr.Markdown("### Current Instructions")
                    inst_summary_btn = gr.Button("Refresh Summary")
                    inst_summary = gr.Markdown()
                    
                    inst_summary_btn.click(
                        self.get_instruction_summary,
                        outputs=inst_summary
                    )
                
                # Web Scraping Tab
                with gr.Tab("🌐 Web Scraping"):
                    gr.Markdown("### Scrape websites for training data")
                    gr.Markdown("⚠️ **Note:** Scraping respects robots.txt and implements rate limiting")
                    
                    urls_input = gr.Textbox(
                        label="URLs to Scrape (one per line)",
                        placeholder="https://example.com\nhttps://another-site.com",
                        lines=5
                    )
                    max_pages_input = gr.Slider(
                        label="Maximum Pages",
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1
                    )
                    
                    with gr.Row():
                        scrape_btn = gr.Button("Start Scraping", variant="primary")
                        clear_data_btn = gr.Button("Clear Scraped Data")
                    
                    scrape_output = gr.Textbox(label="Scraping Status", lines=2)
                    
                    scrape_btn.click(
                        self.start_scraping,
                        inputs=[urls_input, max_pages_input],
                        outputs=scrape_output
                    )
                    clear_data_btn.click(
                        self.clear_scraped_data,
                        outputs=scrape_output
                    )
                
                # Training Tab
                with gr.Tab("🎓 Training"):
                    gr.Markdown("### Train the model on scraped data")
                    
                    epochs_input = gr.Slider(
                        label="Number of Epochs",
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1
                    )
                    
                    train_btn = gr.Button("Start Training", variant="primary")
                    train_output = gr.Textbox(label="Training Status", lines=2)
                    
                    train_btn.click(
                        self.start_training,
                        inputs=epochs_input,
                        outputs=train_output
                    )
                
                # Status Tab
                with gr.Tab("📊 Status"):
                    gr.Markdown("### System Status and Statistics")
                    
                    status_btn = gr.Button("Refresh Status")
                    status_output = gr.Markdown()
                    
                    status_btn.click(
                        self.get_status,
                        outputs=status_output
                    )
                    
                    # Auto-refresh status every 5 seconds
                    interface.load(
                        self.get_status,
                        outputs=status_output,
                        every=5
                    )
            
            gr.Markdown("---")
            gr.Markdown("**Matt AI** - Self-improving AI with ethical standards | Built with ❤️")
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Arguments to pass to gr.Interface.launch()
        """
        interface = self.create_interface()
        
        self.logger.info("Launching chat interface...")
        
        interface.launch(**kwargs)
