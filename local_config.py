# Конфигурация для локальной Llama
LOCAL_LLM_CONFIG = {
    "enabled": True,
    "api_url": "http://localhost:11434/v1",
    "model": "llama3.2:3b", 
    "timeout": 10,
    "max_tokens": 1000,
    "temperature": 0.7
}

# Fallback на OpenRouter если локальная не работает  
OPENROUTER_CONFIG = {
    "enabled": True,
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "model": "meta-llama/llama-3.3-8b-instruct:free",
    "timeout": 30
}

# Настройки приоритетов
PRIORITY_CONFIG = {
    "use_local_first": True,
    "auto_fallback": True,
    "retry_local_after_failure": True
}
