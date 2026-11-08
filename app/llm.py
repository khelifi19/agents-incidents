from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

from typing import Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

import os
llm = init_chat_model("qwen2.5:7b", model_provider="ollama", temperature=0,
                      base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
# 2. Les répliques de la pièce
messages=[
    SystemMessage("Tu es un analyste d'incidents dans une banque. "
        "Réponds en 2 phrases maximum, avec un ton professionnel."),
    HumanMessage("Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle. "
        "Est-ce grave ?")
]

class Classification(BaseModel):
    """Classification d'un incident IT bancaire."""
    categorie: Literal["réseau", "base de données", "applicatif", "sécurité", "hors périmètre"]
    criticite: int = Field(ge=1, le=4, description="1 = mineur, 4 = clients ou argent bloqués")
    service: Literal["SEPA", "Paiement", "Carte", "Appli mobile", "Autre"]
    resume: str = Field(description="UNE seule phrase courte")

CONSIGNES = """Tu classes des incidents informatiques dans une banque.

Catégories :
- réseau : connectivité, load balancer, DNS, pare-feu, lenteur réseau
- base de données : Oracle, SQL, requêtes lentes ou en timeout, verrous
- applicatif : bug dans une application, écran, bouton, page qui ne s'affiche pas
- sécurité : intrusion, connexions suspectes, fuite de données, virus
- hors périmètre : tout ce qui n'est pas un incident informatique

Criticité : 4 = clients ou argent bloqués ; 3 = service dégradé ou risque important ;
2 = gêne limitée ; 1 = mineur.

Exemples :
- "Les requêtes sur la base clients prennent 30 secondes" -> base de données, 3
- "Erreur 500 sur l'écran de connexion de l'appli web" -> applicatif, 3"""

prompt = ChatPromptTemplate.from_messages([
    ("system", CONSIGNES),
    ("human", "{incident}"),
])

classifieur = prompt | llm.with_structured_output(Classification)

if __name__ == "__main__":
    incidents = [
        "Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle.",
        "Le load balancer du site de paiement ne répond plus, aucun client ne peut payer.",
        "Tentatives de connexion suspectes répétées sur le serveur de l'application Carte.",
        "Le bouton 'Exporter en PDF' de l'appli mobile affiche une page blanche.",
        "Le café de la machine du 3e étage est froid.",
    ]
    for texte in incidents:
        res = classifieur.invoke({"incident": texte})
        print(type(res).__name__, "->", res.categorie, "| criticité", res.criticite, "|", res.service)
        print("   ", res.model_dump())
