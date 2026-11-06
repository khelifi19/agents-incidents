from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

# 1. Relier la télécommande au perroquet (modèle local, fantaisie minimale)
llm = init_chat_model("qwen2.5:7b", model_provider="ollama", temperature=0)
# 2. Les répliques de la pièce
messages=[
   
    HumanMessage("Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle. "
        "Est-ce grave ?")
]

if __name__ == "__main__": 
      # 3. La lettre : réponse complète
    response = llm.invoke(messages)
    print("Réponse :", response.content)
    print("Type :", type(response).__name__)
    print("Lego utilisés :", response.usage_metadata)   
     # 4. Le téléphone : réponse mot par mot

    print("\nEn direct : ", end="")
    for token in llm.stream(messages):
        print(token.content, end="", flush=True)
    print()
