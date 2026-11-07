from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from app.llm import llm

PROCEDURES = {
    "base de données": "Basculer sur la base secondaire, puis ouvrir un ticket DBA.",
    "réseau": "Vérifier le load balancer et redémarrer la passerelle concernée.",
    "applicatif": "Revenir à la version précédente (rollback) et prévenir l'équipe projet.",
    "sécurité": "Isoler le serveur et prévenir immédiatement le SOC.",
}
INCIDENTS = {"SEPA": 12, "Paiement": 3, "Carte": 5, "Appli mobile": 7}

@tool
def compter_incidents(service: str) -> int:
    """Renvoie le nombre d'incidents ouverts cette semaine pour un service.
    Valeurs possibles : 'SEPA', 'Paiement', 'Carte', 'Appli mobile'."""
    return INCIDENTS.get(service, 0)

@tool
def rechercher_procedure(categorie: str) -> str:
    """Renvoie la procédure interne à suivre pour une catégorie d'incident.
    Valeurs possibles : 'réseau', 'base de données', 'applicatif', 'sécurité'."""
    return PROCEDURES.get(categorie, "Aucune procédure trouvée.")

outils = [compter_incidents, rechercher_procedure]
outils_par_nom = {o.name: o for o in outils} 

llm_avec_outils = llm.bind_tools(outils)

if __name__ == "__main__":
     # A. Les étiquettes : ce que voit le perroquet
    for o in outils:
        print(o.name, "|", o.args)

    messages = [
        HumanMessage("Combien d'incidents ont été ouverts pour SEPA cette semaine ?"
         "et quelle procédure suivre pour un problème de base de données ?")]

    ai = llm_avec_outils.invoke(messages)
    messages.append(ai)

    tour = 1
    while ai.tool_calls and tour <= 5:         # tant qu'il lève la main (5 tours max)
        print(f"\nTour {tour} - demandes :", [(a["name"], a["args"]) for a in ai.tool_calls])
        for appel in ai.tool_calls:
            outil = outils_par_nom[appel["name"]]
            resultat = outil.invoke(appel["args"])
            messages.append(ToolMessage(str(resultat), tool_call_id=appel["id"]))
        ai = llm_avec_outils.invoke(messages)  # on lui redonne la parole
        messages.append(ai)
        tour += 1

    print("\nRéponse finale :", ai.content)