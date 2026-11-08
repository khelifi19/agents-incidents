import uuid
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage
from langgraph.types import Command

from app.graph import graphe

api = FastAPI(title="Agent d'analyse d'incidents")

# La douane : ce qu'on accepte en entrée
class NouvelIncident(BaseModel):
    incident: str = Field(min_length=15, max_length=2000, description="Description de l'incident")

class Decision(BaseModel):
    decision: Literal["oui", "non"]

def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

def etat(thread_id: str) -> dict:
    """La photo du dossier, présentée proprement."""
    s = graphe.get_state(config(thread_id))
    if not s.values:
        raise HTTPException(404, "Incident inconnu")
    return {
        "thread_id": thread_id,
        "statut": "en_attente_validation" if s.next else "termine",
        "classification": s.values.get("classification"),
        "decision": s.values.get("decision"),
        "dernier_message": s.values["messages"][-1].content,
    }

@api.post("/incidents")
def analyser(body: NouvelIncident):
    thread_id = str(uuid.uuid4())                      # un nouveau numéro de dossier
    graphe.invoke({"messages": [HumanMessage(body.incident)]}, config(thread_id))
    return etat(thread_id)                             # s'arrête à la validation

@api.post("/incidents/{thread_id}/decision")
def decider(thread_id: str, body: Decision):
    s = graphe.get_state(config(thread_id))
    if not s.values:
        raise HTTPException(404, "Incident inconnu")
    if not s.next:
        raise HTTPException(409, "Aucune validation en attente pour cet incident")
    graphe.invoke(Command(resume=body.decision), config(thread_id))
    return etat(thread_id)

@api.get("/incidents/{thread_id}")
def consulter(thread_id: str):
    return etat(thread_id)