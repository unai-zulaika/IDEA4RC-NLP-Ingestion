"""
VLLM Runner Module

Provides VLLM backend integration for LLM inference.
Supports batch inference and falls back to llama.cpp if VLLM is unavailable.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Global VLLM client state
_VLLM_CLIENT: Optional['VLLMClient'] = None
_USE_VLLM: bool = False


class VLLMClient:
    """Client for VLLM server API."""
    
    def __init__(self, endpoint: str, model_name: str, timeout: int = 30):
        """
        Initialize VLLM client.
        
        Args:
            endpoint: VLLM server endpoint (e.g., "http://localhost:8000/v1")
            model_name: Model name to use
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to VLLM server."""
        try:
            # VLLM OpenAI-compatible API uses /v1/models endpoint
            # Remove /v1 from endpoint if present, then add /v1/models
            base_endpoint = self.endpoint.rstrip('/')
            if base_endpoint.endswith('/v1'):
                base_endpoint = base_endpoint[:-3]
            
            response = self.session.get(
                f"{base_endpoint}/v1/models",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json()
                print(f"[VLLM] Connected to server at {self.endpoint}")
                if 'data' in models and len(models['data']) > 0:
                    available_models = [m.get('id', 'unknown') for m in models['data']]
                    print(f"[VLLM] Available models: {', '.join(available_models)}")
            else:
                raise ConnectionError(f"VLLM server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to VLLM server at {self.endpoint}: {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to VLLM server at {self.endpoint}: {e}")
    def generate(self, 
                 prompt: str,
                 max_new_tokens: int = 128,
                 temperature: float = 0.1,
                 **kwargs) -> Dict[str, str]:
        """
        Generate text using VLLM server.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with 'raw' and 'normalized' output
        """
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise medical text annotation assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            raw_output = result["choices"][0]["message"]["content"]
            
            # Normalize: extract first line
            first_line = raw_output.strip().splitlines()[0].strip()
            
            return {
                "raw": raw_output,
                "normalized": first_line
            }
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"VLLM API request failed: {e}")
    
    def generate_batch(self,
                       prompts: List[str],
                       max_new_tokens: int = 128,
                       temperature: float = 0.1,
                       **kwargs) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            List of dictionaries with 'raw' and 'normalized' output
        """
        url = f"{self.endpoint}/chat/completions"
        
        # Prepare batch requests
        results = []
        for prompt in prompts:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise medical text annotation assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                raw_output = result["choices"][0]["message"]["content"]
                first_line = raw_output.strip().splitlines()[0].strip()
                
                results.append({
                    "raw": raw_output,
                    "normalized": first_line
                })
            except requests.exceptions.RequestException as e:
                # Return error result for this prompt
                results.append({
                    "raw": f"ERROR: {str(e)}",
                    "normalized": f"ERROR: {str(e)}"
                })
        
        return results


def load_vllm_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load VLLM configuration from file or environment variables.
    
    Args:
        config_path: Path to config file (optional)
    
    Returns:
        Configuration dictionary
    """
    # Check environment variables first
    use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
    vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    model_name = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    batch_size = int(os.getenv("VLLM_BATCH_SIZE", "8"))
    timeout = int(os.getenv("VLLM_TIMEOUT", "30"))
    
    # Override with config file if provided
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                use_vllm = config.get("use_vllm", use_vllm)
                vllm_endpoint = config.get("vllm_endpoint", vllm_endpoint)
                model_name = config.get("model_name", model_name)
                batch_size = config.get("batch_size", batch_size)
                timeout = config.get("timeout", timeout)
        except Exception as e:
            print(f"[WARN] Failed to load VLLM config from {config_path}: {e}")
    
    return {
        "use_vllm": use_vllm,
        "vllm_endpoint": vllm_endpoint,
        "model_name": model_name,
        "batch_size": batch_size,
        "timeout": timeout
    }


def init_model_vllm(config_path: Optional[Path] = None) -> bool:
    """
    Initialize VLLM client.
    
    Args:
        config_path: Path to VLLM config file (optional)
    
    Returns:
        True if VLLM was successfully initialized, False otherwise
    """
    global _VLLM_CLIENT, _USE_VLLM
    
    config = load_vllm_config(config_path)
    
    if not config["use_vllm"]:
        print("[VLLM] VLLM disabled in configuration")
        return False
    
    try:
        _VLLM_CLIENT = VLLMClient(
            endpoint=config["vllm_endpoint"],
            model_name=config["model_name"],
            timeout=config["timeout"]
        )
        _USE_VLLM = True
        print(f"[VLLM] Initialized successfully with model: {config['model_name']}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to initialize VLLM: {e}")
        print("[WARN] Falling back to llama.cpp")
        _USE_VLLM = False
        _VLLM_CLIENT = None
        return False


def run_model_with_prompt_vllm(prompt: str,
                               max_new_tokens: int = 128,
                               temperature: float = 0.1) -> Dict[str, str]:
    """
    Run model with prompt using VLLM backend.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Dictionary with 'raw' and 'normalized' output
    """
    global _VLLM_CLIENT
    
    if _VLLM_CLIENT is None:
        raise RuntimeError("VLLM client not initialized. Call init_model_vllm() first.")
    
    return _VLLM_CLIENT.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )


def is_vllm_available() -> bool:
    """Check if VLLM is available and initialized."""
    return _USE_VLLM and _VLLM_CLIENT is not None
