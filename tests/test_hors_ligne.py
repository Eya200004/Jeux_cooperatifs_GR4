"""Tests des outils @tool avec .invoke(), sans LLM (étiquette « scripté »)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
pytest.importorskip("langchain_core")
from outils_cooperatifs import coeur, shapley_vote, valeur_shapley

COURS = {"1,2": 60, "1,3": 60, "2,3": 0, "1,2,3": 90}


def test_valeur_shapley_exact():
    r = json.loads(valeur_shapley.invoke({"joueurs": ["1", "2", "3"], "v": COURS}))
    assert r["shapley"] == {"1": 50.0, "2": 20.0, "3": 20.0}


def test_valeur_shapley_mc():
    r = json.loads(valeur_shapley.invoke(
        {"joueurs": ["1", "2", "3"], "v": COURS, "methode": "mc", "n_perm": 5000, "seed": 0}))
    assert abs(r["shapley"]["1"] - 50) < 2
    assert "erreur_standard" in r


def test_coeur_non_vide():
    r = json.loads(coeur.invoke({"joueurs": ["1", "2", "3"], "v": COURS}))
    assert r["coeur_vide"] is False
    assert r["bornes"]["1"] == {"minimum": 30.0, "maximum": 90.0}


def test_coeur_vide_majorite():
    v = {"1,2": 1, "1,3": 1, "2,3": 1, "1,2,3": 1}
    r = json.loads(coeur.invoke({"joueurs": ["1", "2", "3"], "v": v}))
    assert r["coeur_vide"] is True


def test_shapley_vote():
    r = json.loads(shapley_vote.invoke({"quota": 51, "poids": [40, 30, 20, 10]}))
    assert r["indices"] == pytest.approx([5 / 12, 1 / 4, 1 / 4, 1 / 12], abs=1e-5)


def test_plafond_n():
    r = json.loads(shapley_vote.invoke({"quota": 11, "poids": [1] * 21}))
    assert "erreur" in r


def test_joueur_inconnu():
    with pytest.raises(ValueError):
        coeur.invoke({"joueurs": ["1", "2"], "v": {"1,9": 5}})
