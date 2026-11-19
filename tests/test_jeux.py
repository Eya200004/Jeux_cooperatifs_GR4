"""Tests des fonctions pures documentées dans src/jeux.py."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jeux import coeur_lp, shapley_exact, shapley_mc, shapley_shubik


def jeu_du_cours(mask):
    valeurs = {0b011: 60, 0b101: 60, 0b111: 90}
    return float(valeurs.get(mask, 0))


def majorite(mask):
    return 1.0 if mask.bit_count() >= 2 else 0.0


def test_shapley_exact_jeu_du_cours():
    assert shapley_exact(jeu_du_cours, 3) == pytest.approx([50, 20, 20])


def test_shapley_mc_approche_exact_et_erreurs_non_negatives():
    phi, erreurs = shapley_mc(jeu_du_cours, 3, K=20000, seed=0)
    assert phi == pytest.approx([50, 20, 20], abs=1.0)
    assert all(erreur >= 0 for erreur in erreurs)


def test_shapley_mc_reproductible_avec_la_graine():
    assert shapley_mc(jeu_du_cours, 3, K=500, seed=42) == shapley_mc(
        jeu_du_cours, 3, K=500, seed=42
    )


def test_coeur_retourne_les_bornes_connues():
    bornes = coeur_lp(jeu_du_cours, 3)
    assert bornes is not None
    for resultat, attendu in zip(bornes, [(30, 90), (0, 30), (0, 30)]):
        assert resultat == pytest.approx(attendu)


def test_coeur_vide_pour_le_jeu_majoritaire():
    assert coeur_lp(majorite, 3) is None


def test_shapley_shubik_vote_pondere():
    poids = [40, 30, 20, 10]
    quota = 51

    def vote(mask):
        total = sum(
            poids[i] for i in range(len(poids)) if mask & (1 << i)
        )
        return 1.0 if total >= quota else 0.0

    assert shapley_shubik(vote, len(poids)) == pytest.approx(
        [5 / 12, 1 / 4, 1 / 4, 1 / 12]
    )


def test_shapley_shubik_rejette_un_jeu_non_monotone():
    def non_monotone(mask):
        return 1.0 if mask in (0b001, 0b111) else 0.0

    with pytest.raises(ValueError, match="monotone"):
        shapley_shubik(non_monotone, 3)


def test_shapley_mc_rejette_un_nombre_de_permutations_insuffisant():
    with pytest.raises(ValueError, match="K"):
        shapley_mc(jeu_du_cours, 3, K=1)
