"""Fonctions pures pour les jeux coopératifs (sans LangChain).

Conventions :
- Les joueurs sont numérotés 0 à n-1.
- Une coalition est un bitmask (entier) : le bit i vaut 1 si le joueur i
  appartient à la coalition. Le bitmask 0 est la coalition vide.
- v est la fonction caractéristique : v(mask) -> float, avec v(0) = 0.
"""

from math import factorial

import numpy as np
from scipy.optimize import linprog


def _valeurs(n, v):
    """Tableau des v(mask) pour tous les masks de 0 à 2^n - 1."""
    return np.array([float(v(m)) for m in range(1 << n)])


def shapley_exact(n, v):
    """Valeur de Shapley exacte par coalitions (bitmasks).

    Retourne un tableau numpy de taille n avec phi_i pour chaque joueur.
    Coût en O(2^n) : raisonnable jusqu'à n ≈ 20.
    """
    vals = _valeurs(n, v)
    masks = np.arange(1 << n)
    taille = np.array([bin(m).count("1") for m in masks])
    fact = np.array([factorial(k) for k in range(n + 1)], dtype=float)
    phi = np.zeros(n)
    for i in range(n):
        sans_i = masks[(masks >> i) & 1 == 0]
        poids = fact[taille[sans_i]] * fact[n - taille[sans_i] - 1] / fact[n]
        gain = vals[sans_i | (1 << i)] - vals[sans_i]
        phi[i] = np.sum(poids * gain)
    return phi


def shapley_mc(n, v, n_perm, seed):
    """Estimation de Shapley par permutations aléatoires.

    Retourne (phi, erreur_standard), deux tableaux numpy de taille n.
    La graine seed rend le résultat reproductible.
    """
    rng = np.random.default_rng(seed)
    somme = np.zeros(n)
    somme_carres = np.zeros(n)
    for _ in range(n_perm):
        ordre = rng.permutation(n)
        mask = 0
        v_prec = float(v(0))
        for i in ordre:
            mask |= 1 << int(i)
            v_new = float(v(mask))
            contrib = v_new - v_prec
            somme[i] += contrib
            somme_carres[i] += contrib ** 2
            v_prec = v_new
    phi = somme / n_perm
    variance = np.maximum(somme_carres / n_perm - phi ** 2, 0.0)
    erreur = np.sqrt(variance / n_perm)
    return phi, erreur


def coeur_lp(n, v, cout=False):
    """Cherche un point du cœur avec linprog.

    Jeu de gains : sum(x) = v(N) et sum_{i in S} x_i >= v(S) pour tout S.
    Jeu de coûts (cout=True) : sum(x) = v(N) et sum_{i in S} x_i <= v(S).
    Retourne un tableau numpy, ou None si le cœur est vide (système
    infaisable). Lève RuntimeError si le solveur échoue pour une autre
    raison (statut différent de 0 et 2).
    """
    grande = (1 << n) - 1
    lignes, seconds = [], []
    for mask in range(1, grande):
        ligne = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        if cout:
            lignes.append(ligne)
            seconds.append(float(v(mask)))
        else:
            lignes.append(-ligne)
            seconds.append(-float(v(mask)))
    res = linprog(
        c=np.zeros(n),
        A_ub=np.array(lignes) if lignes else None,
        b_ub=np.array(seconds) if seconds else None,
        A_eq=np.ones((1, n)),
        b_eq=[float(v(grande))],
        bounds=[(None, None)] * n,
        method="highs",
    )
    if res.status == 0:
        return res.x
    if res.status == 2:
        return None
    raise RuntimeError(f"Échec du solveur linprog (statut {res.status}) : {res.message}")


def shapley_shubik(quota, poids):
    """Indices de Shapley-Shubik d'un vote pondéré [quota ; w1, ..., wn].

    v(S) = 1 si la somme des poids de S >= quota, sinon 0.
    Retourne un tableau numpy de taille n (somme = 1).
    """
    n = len(poids)

    def v(mask):
        total = sum(poids[i] for i in range(n) if (mask >> i) & 1)
        return 1.0 if total >= quota else 0.0

    return shapley_exact(n, v)


def dans_coeur(n, v, x, cout=False, tol=1e-9):
    """Vérifie si l'allocation x appartient au cœur du jeu."""
    grande = (1 << n) - 1
    x = np.asarray(x, dtype=float)
    if abs(x.sum() - float(v(grande))) > tol:
        return False
    for mask in range(1, grande):
        part = sum(x[i] for i in range(n) if (mask >> i) & 1)
        if cout and part > float(v(mask)) + tol:
            return False
        if not cout and part < float(v(mask)) - tol:
            return False
    return True


def jeu_depuis_dict(valeurs):
    """Construit v(mask) depuis un dict {frozenset de joueurs 0-indexés: valeur}.

    Les coalitions absentes valent 0.
    """
    def v(mask):
        coalition = frozenset(i for i in range(mask.bit_length()) if (mask >> i) & 1)
        return float(valeurs.get(coalition, 0.0))
    return v
