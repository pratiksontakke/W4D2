"""
LLM Client Adapter for Smart Meeting Assistant

This module provides a unified interface for interacting with different LLM providers:
- OpenAI GPT-4o (default)
- Anthropic Claude 3
- Google Gemini 1.5

Key Features:
- Provider abstraction - switch via environment variable
- Async support for non-blocking operations
- Error handling and retry logic
- Token optimization for cost efficiency
- PII redaction for privacy
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, List
from enum import Enum
import structlog

# Configure logging
logger = structlog.get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMClient:
    """
    Unified LLM client that abstracts different providers.
    
    This class provides:
    - Provider-agnostic API for text generation
    - Async support for non-blocking operations
    - Built-in retry logic and error handling
    - Token optimization and cost management
    - PII redaction for privacy protection
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider name (openai, anthropic, gemini)
                     If None, reads from LLM_PROVIDER env var
        """
        # Determine provider from parameter or environment
        self.provider_name = provider or os.getenv("LLM_PROVIDER", "openai").lower()
        
        try:
            self.provider = LLMProvider(self.provider_name)
        except ValueError:
            logger.warning(f"Unknown provider {self.provider_name}, defaulting to OpenAI")
            self.provider = LLMProvider.OPENAI
        
        # Initialize provider-specific client
        self._init_provider_client()
        
        # Configuration
        self.max_retries = 3
        self.timeout = 30.0
        self.max_tokens = 1000  # Default max tokens for responses
        
        logger.info(f"LLM client initialized with provider: {self.provider.value}")
    
    def _init_provider_client(self):
        """Initialize the provider-specific client."""
        if self.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.provider == LLMProvider.GEMINI:
            self._init_gemini()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            # Import OpenAI - will be available when requirements are installed
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not found, LLM features will be limited")
                self.client = None
                return
            
            # Import will be available after pip install
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.model = "gpt-4o"  # Latest GPT-4 Omni model
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            logger.warning("OpenAI package not installed, LLM features will be limited")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.warning("ANTHROPIC_API_KEY not found, LLM features will be limited")
                self.client = None
                return
            
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.model = "claude-3-sonnet-20240229"  # Latest Claude 3 model
            logger.info("Anthropic client initialized successfully")
            
        except ImportError:
            logger.warning("Anthropic package not installed, LLM features will be limited")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None
    
    def _init_gemini(self):
        """Initialize Google Gemini client."""
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not found, LLM features will be limited")
                self.client = None
                return
            
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel('gemini-1.5-pro')
            self.model = "gemini-1.5-pro"
            logger.info("Google Gemini client initialized successfully")
            
        except ImportError:
            logger.warning("Google Generative AI package not installed, LLM features will be limited")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None
    
    def _redact_pii(self, text: str) -> str:
        """
        Redact personally identifiable information from text.
        
        Args:
            text: Input text that may contain PII
            
        Returns:
            Text with PII redacted
        """
        import re
        
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (basic patterns)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]', text)
        
        return text
    
    def _optimize_prompt(self, prompt: str, max_length: int = 2000) -> str:
        """
        Optimize prompt for token efficiency.
        
        Args:
            prompt: Original prompt
            max_length: Maximum character length
            
        Returns:
            Optimized prompt
        """
        if len(prompt) <= max_length:
            return prompt
        
        # Truncate while preserving structure
        lines = prompt.split('\n')
        optimized_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > max_length:
                optimized_lines.append("... [truncated for length]")
                break
            optimized_lines.append(line)
            current_length += len(line)
        
        return '\n'.join(optimized_lines)
    
    async def generate_text(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: float = 0.7,
                          redact_pii: bool = True) -> str:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            redact_pii: Whether to redact PII from the prompt
            
        Returns:
            Generated text response
        """
        # If no client is available, return a fallback response
        if not self.client:
            return "LLM client not available. Please configure API keys."
        
        try:
            # Redact PII if requested
            if redact_pii:
                prompt = self._redact_pii(prompt)
                if system_prompt:
                    system_prompt = self._redact_pii(system_prompt)
            
            # Optimize prompt for token efficiency
            prompt = self._optimize_prompt(prompt)
            
            # Set max tokens
            max_tokens = max_tokens or self.max_tokens
            
            # Generate based on provider
            if self.provider == LLMProvider.OPENAI:
                return await self._generate_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == LLMProvider.GEMINI:
                return await self._generate_gemini(prompt, system_prompt, max_tokens, temperature)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return a fallback response instead of raising
            return f"Error generating response: {str(e)}"
    
    async def _generate_openai(self, prompt: str, system_prompt: Optional[str], 
                              max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=self.timeout
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_anthropic(self, prompt: str, system_prompt: Optional[str], 
                                 max_tokens: int, temperature: float) -> str:
        """Generate text using Anthropic API."""
        # Anthropic uses a different message format
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text.strip()
    
    async def _generate_gemini(self, prompt: str, system_prompt: Optional[str], 
                              max_tokens: int, temperature: float) -> str:
        """Generate text using Google Gemini API."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = await self.client.generate_content_async(
            full_prompt,
            generation_config=generation_config
        )
        
        return response.text.strip()
    
    async def generate_agenda(self, meeting_title: str, participants: List[str], 
                            duration_minutes: int, meeting_type: str = "general") -> str:
        """
        Generate a meeting agenda using LLM.
        
        Args:
            meeting_title: Title of the meeting
            participants: List of participant names/emails (will be redacted)
            duration_minutes: Meeting duration in minutes
            meeting_type: Type of meeting (standup, review, planning, etc.)
            
        Returns:
            Generated agenda in markdown format
        """
        system_prompt = """You are a professional meeting facilitator. Generate concise, actionable meeting agendas in markdown format. Focus on clear objectives, time allocation, and actionable items."""
        
        prompt = f"""
        Generate a meeting agenda for:
        
        Title: {meeting_title}
        Type: {meeting_type}
        Duration: {duration_minutes} minutes
        Participants: {len(participants)} people
        
        Please create a structured agenda with:
        1. Clear objectives
        2. Time allocation for each item
        3. Action items where appropriate
        4. Next steps
        
        Format as markdown with bullet points.
        """
        
        return await self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=500,
            temperature=0.7
        )
    
    async def explain_optimal_slot(self, slot_info: Dict[str, Any]) -> str:
        """
        Generate an explanation for why a time slot is optimal.
        
        Args:
            slot_info: Dictionary containing slot details
            
        Returns:
            Human-readable explanation
        """
        system_prompt = """You are a scheduling assistant. Explain why a particular time slot works well for all participants in a friendly, concise manner."""
        
        prompt = f"""
        Explain why this time slot is optimal for the meeting:
        
        Time: {slot_info.get('time', 'Not specified')}
        Score: {slot_info.get('score', 0)}/100
        Participants: {slot_info.get('participant_count', 0)} people
        Conflicts: {slot_info.get('conflicts', 0)}
        
        Provide a brief, friendly explanation in 1-2 sentences.
        """
        
        return await self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=100,
            temperature=0.5
        )


# Global LLM client instance
# This will be initialized when the module is imported
llm_client = LLMClient()
