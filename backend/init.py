
import logging
import dotenv
import os
import openai
from typing import List, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

openai_model = "gpt-4.1"

openai_client = openai.OpenAI(api_key=openai_api_key)
