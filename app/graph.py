from typing import Annotated, Optional, TypedDict
from langchain.messages import SystemMessage, HumanMessage ,AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.llm import classifieur


from langgraph.prebuilt import ToolNode, tools_condition
from app.tools import llm_avec_outils, outils

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# 1. Le dossier qui circule
class State(TypedDict):
    messages: Annotated[list, add_messages]          # règle : on AJOUTE à la suite
    classification: Optional[dict]   
    decision: Optional[str]         

# 2. Les postes de travail
def classer(state: State):
    if state.get("classification"):
           return {}
    texte = state["messages"][-1].content
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


def validation_humaine(state: State):
    print("[validation] la pause commence")      # volontairement AVANT interrupt (expérience)
    decision = interrupt({
        "action_proposee": state["messages"][-1].content,
        "question": "Approuver cette action ? (oui/non)",
    })
    return {"decision": decision}

def executer(state: State):
    print("[executer] action exécutée")
    return {"messages": [AIMessage("Action enregistrée dans l'outil de ticketing (simulation).")]}

def apres_validation(state: State) -> str:
    return "executer" if state["decision"] == "oui" else END

# 3. Le plan de l'usine
g = StateGraph(State)
g.add_node("classer", classer)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(outils))  
g.add_node("validation_humaine", validation_humaine)
g.add_node("executer", executer)
g.add_edge(START, "classer")
g.add_edge("classer", "agent")

# agent : outil demandé -> tools ; sinon -> validation (au lieu de END)
g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "validation_humaine"})
g.add_conditional_edges("validation_humaine", apres_validation, ["executer", END])
g.add_edge("executer", END)
g.add_edge("tools", "agent")

graphe = g.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    dossier = {"configurable": {"thread_id": "incident-SEPA-010"}}

    # 1. L'usine tourne jusqu'à la pause
    r = graphe.invoke({"messages": [HumanMessage(
        "Depuis 9h, les virements SEPA échouent avec un timeout sur la base Oracle.")]}, dossier)

    pause = r["__interrupt__"][0].value
    print("\nAction proposée :", pause["action_proposee"])
    print("Poste en attente :", graphe.get_state(dossier).next)

    # 2. Le chef décide
    reponse = input(pause["question"] + " ").strip().lower()

    # 3. On reprend, MÊME dossier
    final = graphe.invoke(Command(resume=reponse), dossier)
    print("\nDécision :", final["decision"])
    print("Dernier message :", final["messages"][-1].content)