
import json
import os
import sys

import numpy as np
from langchain_core.tools import tool

# Accès au dossier src
SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from jeux import coeur_lp, shapley_exact, shapley_mc, shapley_shubik

N_MAX_EXACT = 20
N_MAX_COEUR = 15


def _json(x):
    """Convertit les types NumPy en types Python, puis en JSON."""
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    return json.dumps(x, default=conv, ensure_ascii=False)


def _construire_v(joueurs, v):
    """Transforme les coalitions en fonction caractéristique v(mask)."""
    noms = [str(j) for j in joueurs]
    table = {}

    for cle, valeur in v.items():
        membres = frozenset(
            t.strip() for t in str(cle).split(",") if t.strip()
        )
        inconnus = membres - set(noms)

        if inconnus:
            raise ValueError(
                f"Joueur inconnu dans la coalition {cle} : {sorted(inconnus)}"
            )

        table[membres] = float(valeur)

    def fv(mask):
        coalition = frozenset(
            noms[i]
            for i in range(len(noms))
            if (mask >> i) & 1
        )
        return table.get(coalition, 0.0)

    return noms, fv


@tool
def coeur(joueurs: list[str], v: dict[str, float]) -> str:
    """Cherche les bornes des allocations du cœur d'un jeu coopératif."""
    n = len(joueurs)

    if n > N_MAX_COEUR:
        return _json({
            "erreur": f"n = {n} dépasse le plafond {N_MAX_COEUR}"
        })

    noms, fv = _construire_v(joueurs, v)

    # Signature attendue : coeur_lp(v, n)
    resultats = coeur_lp(fv, n)

    if resultats is None:
        return _json({"coeur_vide": True, "bornes": None})

    bornes = {
        noms[i]: {
            "minimum": round(float(resultats[i][0]), 6),
            "maximum": round(float(resultats[i][1]), 6)
        }
        for i in range(n)
    }

    return _json({"coeur_vide": False, "bornes": bornes})


@tool
def shapley_vote(quota: float, poids: list[float]) -> str:
    """Calcule les indices de pouvoir de Shapley-Shubik."""
    n = len(poids)

    if n > N_MAX_EXACT:
        return _json({"erreur": f"n = {n} dépasse le plafond {N_MAX_EXACT}"})

    if n == 0 or quota <= 0 or any(p < 0 for p in poids):
        return _json({"erreur": "Quota ou poids invalides"})

    def v_vote(mask):
        total = sum(
            poids[i]
            for i in range(n)
            if (mask >> i) & 1
        )
        return 1.0 if total >= quota else 0.0

    ssi = shapley_shubik(v_vote, n)

    return _json({
        "indices": [
            round(float(s), 6) for s in ssi
        ]
    })


@tool
def valeur_shapley(
    joueurs: list[str],
    v: dict[str, float],
    methode: str = "exact",
    n_perm: int = 2000,
    seed: int = 0
) -> str:
    """Calcule la valeur de Shapley exacte ou par Monte-Carlo."""
    n = len(joueurs)

    if n == 0:
        return _json({"erreur": "Il faut au moins un joueur"})

    noms, fv = _construire_v(joueurs, v)

    if methode == "exact":
        if n > N_MAX_EXACT:
            return _json({
                "erreur": f"n = {n} dépasse le plafond {N_MAX_EXACT}"
            })

        phi = shapley_exact(fv, n)

        return _json({
            "shapley": {
                noms[i]: round(float(phi[i]), 6)
                for i in range(n)
            }
        })

    elif methode == "mc":
        if n_perm < 2:
            return _json({"erreur": "n_perm doit être supérieur ou égal à 2"})

        phi, err = shapley_mc(fv, n, n_perm, seed)

        return _json({
            "shapley": {
                noms[i]: round(float(phi[i]), 6)
                for i in range(n)
            },
            "erreur_standard": {
                noms[i]: round(float(err[i]), 6)
                for i in range(n)
            }
        })

    return _json({"erreur": "Méthode inconnue : choisir exact ou mc"})


OUTILS = [coeur, shapley_vote, valeur_shapley]
