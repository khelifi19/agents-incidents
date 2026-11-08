from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from app.llm import classifieur

client = Client()
NOM = "incidents-classification"

# Le sujet d'examen + le corrigé (incidents jamais vus par le prompt)
EXAMEN = [
    ("Le serveur DNS interne ne répond plus, les agences n'accèdent plus à l'intranet.", "réseau"),
    ("La requête de clôture mensuelle est bloquée par un verrou sur la base PostgreSQL.", "base de données"),
    ("Le bouton 'Valider le virement' de l'appli mobile ne réagit plus.", "applicatif"),
    ("L'antivirus a détecté un malware sur le poste d'un conseiller.", "sécurité"),
    ("La photocopieuse du 2e étage n'a plus de toner.", "hors périmètre"),
    ("Le pare-feu bloque les connexions sortantes vers le partenaire Visa.", "réseau"),
    ("Un fichier de clients a été envoyé par erreur à une adresse externe.", "sécurité"),
    ("L'écran de connexion de la banque en ligne affiche une erreur 500 après la mise à jour.", "applicatif"),
]

# 1. Enregistrer le dataset dans LangSmith (une seule fois)
if not client.has_dataset(dataset_name=NOM):
    client.create_examples(
        dataset_id=client.create_dataset(dataset_name=NOM).id,
        examples=[{"inputs": {"incident": t}, "outputs": {"categorie": c}} for t, c in EXAMEN],
    )

# 2. Ce qu'on évalue : le classifieur
def cible(inputs: dict) -> dict:
    res = classifieur.invoke({"incident": inputs["incident"]})
    return {"categorie": res.categorie}

# 3. Le correcteur
def bonne_categorie(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["categorie"] == reference_outputs["categorie"]

if __name__ == "__main__":
    resultats = client.evaluate(cible, data=NOM, evaluators=[bonne_categorie],
                                experiment_prefix="prompt-sans-exemples")
    scores = [r["evaluation_results"]["results"][0].score for r in resultats]
    print(f"\nScore : {sum(scores)}/{len(scores)}")