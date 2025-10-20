import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for the review summarizer."""
    
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Model Configuration
    default_model: str = "llama-3.1-8b-instant"
    local_model: str = "flan-t5-base"
    
    # Filtering Parameters
    min_helpful_votes: int = 5
    require_verified: bool = False
    min_review_length: int = 50
    
    # Theme Extraction
    num_themes: int = 10
    theme_min_frequency: float = 0.05
    
    # Summarization
    max_summary_length: int = 120
    chain_of_density_iterations: int = 5
    
    # Evaluation
    rouge_types: list = None
    
    def __post_init__(self):
        if self.rouge_types is None:
            self.rouge_types = ["rouge1", "rouge2", "rougeL"]
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.groq_api_key and not self.openai_api_key:
            print("Warning: No API keys found. Set GROQ_API_KEY or OPENAI_API_KEY in .env file")
            return False
        return True

# Global config instance
config = Config()
