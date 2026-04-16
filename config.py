from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")