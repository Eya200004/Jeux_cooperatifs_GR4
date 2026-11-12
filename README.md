# Projet G4 : Jeux coopératifs

Shapley exact et Monte-Carlo, cœur par programmation linéaire, indice de Shapley-Shubik.

Projet de Théorie des jeux et systèmes multi-agents, Cycle Ingénieur Informatique Décisionnelle et IA.

## Membres et rôles (semaines 1 à 4)

| Membre | Rôle |
|---|---|
| Aya | Chef de projet / Théorie + Testeur |
| Mouaad | Quant / Python |
| Khaoula | Développeur agent |

La rotation des rôles a lieu à la fin de la semaine 4 (voir le Gantt dans `docs/`).

## Installation

1. Python 3.10 ou plus
2. Cloner le dépôt puis installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Installer [Ollama](https://ollama.com) et télécharger le modèle :
   ```bash
   ollama pull llama3.1
   ```

## Lancer

- Tests : `pytest tests/`
- Agent : `python agent_tj_sma.py` (Ollama doit tourner)

## Structure du dépôt

- `docs/` : fiche de lecture, Gantt, points d'avancement, rapport
- `notebooks/` : prototypes et expériences
- `src/jeux.py` : fonctions pures (Shapley exact, Shapley Monte-Carlo, cœur par LP, Shapley-Shubik)
- `src/experiences.py` : protocoles d'expériences
- `agent_tj_sma.py` : agent du cours étendu (outils `coeur`, `shapley_vote`)
- `tests/` : tests pytest

## Convention de commit

`[Prénom] verbe + objet`, par exemple : `[Aya] Ajout du squelette du dépôt`

## Travail en équipe

Chacun travaille sur sa branche (`prenom/sujet`) et fusionne via pull request. Chaque membre doit avoir des commits visibles.