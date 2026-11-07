from typing import Annotated, Optional, TypedDict
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.llm import classifieur


from langgraph.prebuilt import ToolNode, tools_condition
from app.tools import llm_avec_outils, outils

from langgraph.checkpoint.memory import InMemorySaver

# 1. Le dossier qui circule
class State(TypedDict):
    messages: Annotated[list, add_messages]          # règle : on AJOUTE à la suite
    classification: Optional[dict]          # règle par défaut : on remplace

# 2. Les postes de travail
def classer(state: State):
    if state.get("classification"):
           return {}
    texte = state["messages"][0].content
    res = classifieur.invoke({"incident": texte})
    print("[classer] ->", res.categorie, "| criticité", res.criticite)
    return {"classification": res.model_dump()}      # on ne renvoie QUE sa case

def agent(state: State):
    consigne = SystemMessage(
        "Tu es analyste d'incidents dans une banque. "
        f"Classification : {state['classification']}. "
         "Utilise les outils pour compter les incidents du service et trouver la procédure, "
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

graphe = g.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    dossier_a = {"configurable": {"thread_id": "incident-SEPA-001"}}
    dossier_b = {"configurable": {"thread_id": "incident-002"}}

    # 1. Premier message dans le dossier A
    r1 = graphe.invoke({"messages": [HumanMessage(
        "Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle.")]}, dossier_a)
    print("\n[A - 1] :", r1["messages"][-1].content)

    # 2. On revient dans le MÊME dossier : l'agent se souvient-il ?
    r2 = graphe.invoke({"messages": [HumanMessage(
        "Rappelle-moi le service concerné et le nombre d'incidents.")]}, dossier_a)
    print("\n[A - 2] :", r2["messages"][-1].content)

    # 3. Même question, mais dans un AUTRE dossier
    r3 = graphe.invoke({"messages": [HumanMessage(
        "Rappelle-moi le service concerné et le nombre d'incidents.")]},dossier_b)
    print("\n[B - 1] :", r3["messages"][-1].content)

    # 4. La dernière photo du dossier A
    photo = graphe.get_state(dossier_a)
    print("\nDossier A :", len(photo.values["messages"]), "messages | classification :",
          photo.values["classification"]["categorie"])