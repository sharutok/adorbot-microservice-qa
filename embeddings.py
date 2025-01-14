from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from tiktoken import get_encoding

import os
load_dotenv()
def get_embedding_function():
    try:
        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {e}")
