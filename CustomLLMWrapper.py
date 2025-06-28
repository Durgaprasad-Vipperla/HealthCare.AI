import os
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from typing import Dict, Any, Optional, List

load_dotenv()

class CustomLLM(LLM):

    def _call(self, prompt: str, stop: Optional[List[str]] = None, *args, **kwargs) -> str:
        """Send request to FastAPI model and return response."""
        
        headers = {
            "Content_Type": "application/json",
            "Authorization": f"Bearer {os.getenv("CANVAS_BEARER_TOKEN")}"
        }
        
        payload = {
            "query": prompt,
            "space_name": "TrainToDeploy",
            "userId": "938db971-e99d-42b5-b843-4168e7a12e66",
            "temperature": "0.1",
            "type": "chat"
            }
        
        try:
            response = requests.post(os.getenv("CANVAS_API_URL"), json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()

            llm_response = response_json.get("response", "No response received")
            llm_response = str(llm_response).strip()

            return llm_response
        
        except requests.exceptions.RequestException as e:
            print(f"LLM API Error: {str(e)}")
            return f"Error: {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "LLM Wrapper for Canvas API"