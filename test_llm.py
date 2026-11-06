from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
llm=init_chat_model(model="qwen2.5:7b",model_provider="ollama",temperature=0)
print(llm.invoke("Explique ce qu'est un incident informatique en une phrase.").content)

