from typing import Any, Dict, List, Optional, TypeAlias, Union
from pydantic import BaseModel

# Type Alias for key, uid, or date queries



class PromptModel(BaseModel):
    prompt: str
    context: str 
    instruction: Optional[str] = "you are a helpfull bot give responce to the query based on the context given"
    max_tokens: Optional[int] = 800

