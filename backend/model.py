from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

