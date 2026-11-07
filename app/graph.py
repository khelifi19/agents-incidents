from typing import Annotated, Optional, TypedDict
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.llm import classifieur


from langgraph.prebuilt import ToolNode, tools_condition
from app.tools import llm_avec_outils, outils

# 1. Le dossier qui circule
class State(TypedDict):
    messages: Annotated[list, add_messages]          # règle : on AJOUTE à la suite
    classification: Optional[dict]          # règle par défaut : on remplace

# 2. Les postes de travail
def classer(state: State):
    texte = state["messages"][0].content
    res = classifieur.invoke({"incident": texte})
    print("[classer] ->", res.categorie, "| criticité", res.criticite)
    return {"classification": res.model_dump()}      # on ne renvoie QUE sa case

def agent(state: State):
    consigne = SystemMessage(
        "Tu es analyste d'incidents dans une banque. "
        f"Classification : {state['classification']}. "
        "Utilise les outils pour compter les incidents du service et trouver la procédure."
    
        "puis propose UNE action corrective en 2 phrases maximum."
    )
    reponse = llm_avec_outils.invoke([consigne] + state["messages"])
    print("[agent] demandes d'outils :", [(a["name"], a["args"]) for a in reponse.tool_calls])
    return {"messages": [reponse]}

# 3. Le plan de l'usine
g = StateGraph(State)
g.add_node("classer", classer)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(outils))  
g.add_edge(START, "classer")
g.add_edge("classer", "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")

graphe = g.compile()

if __name__ == "__main__":
    print(graphe.get_graph().draw_mermaid())       # le plan, en texte

    resultat = graphe.invoke({"messages": [HumanMessage(
        "Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle.")]})

    print("\nClassification :", resultat["classification"])
    print("Nombre de messages :", len(resultat["messages"]))
    for m in resultat["messages"]:
        print(" -", type(m).__name__, "|", (m.content or "")[:90], "|", getattr(m, "tool_calls", ""))