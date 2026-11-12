# Fiche de lecture – Projet 4 : Jeux coopératifs (Shapley, cœur, indices de pouvoir)

**Groupe 4** – Partie rédigée par Mouaad : Algorithme et hypothèses.
Les définitions suivent la théorie standard des jeux coopératifs ; elles ne reproduisent pas une implémentation particulière du cours.

---

## 1. Cadre et conventions

Un **jeu coopératif à utilité transférable** est un couple `(N, v)` où :

- `N = {0, …, n-1}` est l'ensemble des joueurs ;
- `v : 2^N → ℝ` est la **fonction caractéristique** : `v(S)` est le gain total que la coalition `S` peut s'assurer.

Représentation informatique (identique pour les quatre fonctions de `src/jeux.py`) :

| Objet | Représentation |
|---|---|
| Coalition `S` | entier masque de bits |
| Coalition vide `∅` | `0` |
| Joueur `i` | bit `1 << i` |
| Grande coalition `N` | `(1 << n) - 1` |
| `i ∈ S` | `(S >> i) & 1 == 1` |
| `S ∪ {i}` | `S \| (1 << i)` |
| Fonction caractéristique | `v : Callable[[int], float]` |

Interfaces :

```python
shapley_exact(v, n)         -> list[float]
shapley_mc(v, n, K, seed=None) -> tuple[list[float], list[float]]   # (estimations, erreurs standards)
coeur_lp(v, n)              -> list[tuple[float, float]] | None     # (min, max) par joueur, ou None
shapley_shubik(v, n)        -> list[float]
```

---

## 2. Hypothèses

1. **`v(∅) = 0`** : une coalition vide ne produit aucun gain. Cette normalisation est nécessaire à l'efficacité de la valeur de Shapley.
2. **Utilité transférable** : le gain `v(S)` peut se partager librement entre les membres de `S`, ce qui permet de raisonner sur des sommes de paiements `Σ x_i`.
3. **Superadditivité** : pour deux coalitions disjointes `S` et `T`, `v(S ∪ T) ≥ v(S) + v(T)`. Elle garantit qu'aucune scission n'est avantageuse, donc que la **grande coalition se forme** et que `v(N)` est le gain à répartir.
4. **Calcul exact en `O(n·2ⁿ)`** : exponentiel en `n`, donc raisonnable jusqu'à `n ≈ 20` ; au-delà, on utilise l'estimation Monte-Carlo.
5. **Shapley-Shubik** : défini pour un **jeu simple monotone** (`v(S) ∈ {0, 1}`, `v(N) = 1`), typiquement un vote pondéré.
6. **Cœur** : système écrit pour un **jeu de gains** (inégalités `≥`). Pour un jeu de coûts, les inégalités s'inversent (`≤`).

---

## 3. Fondements théoriques

### 3.1 Valeur de Shapley

La valeur de Shapley du joueur `i` est sa **contribution marginale moyenne** sur tous les ordres d'arrivée :

```
φ_i(v) = (1/n!) · Σ_π [ v(P_i^π ∪ {i}) − v(P_i^π) ]
```

où `P_i^π` est l'ensemble des prédécesseurs de `i` dans la permutation `π`. En regroupant les permutations selon `S = P_i^π`, on obtient la **formule par coalitions** :

```
φ_i(v) = Σ_{S ⊆ N\{i}}  [ |S|! · (n − |S| − 1)! / n! ] · [ v(S ∪ {i}) − v(S) ]
```

Le coefficient `|S|!(n−|S|−1)!/n!` est la proportion des ordres où exactement les joueurs de `S` précèdent `i`.

La valeur de Shapley est **l'unique** répartition vérifiant :

- **efficacité** : `Σ_i φ_i = v(N)` ;
- **symétrie** : deux joueurs interchangeables reçoivent la même valeur ;
- **joueur nul** : un joueur qui n'apporte rien (`v(S ∪ {i}) = v(S)` pour tout `S`) reçoit `0` ;
- **additivité** : `φ(v + w) = φ(v) + φ(w)`.

**Coût.** Le calcul par permutations coûte `O(n·n!)`. La formule par coalitions parcourt `2ⁿ` coalitions et, pour chacune, au plus `n` joueurs : `O(n·2ⁿ)` appels à `v`.

### 3.2 Estimation Monte-Carlo

On tire `K` permutations uniformes `π_1, …, π_K` et on note `m_i^(k)` la contribution marginale de `i` dans `π_k`.

```
φ̂_i = (1/K) Σ_k m_i^(k)
s_i² = (1/(K−1)) Σ_k ( m_i^(k) − φ̂_i )²
SE_i = s_i / √K
```

- L'estimateur est **sans biais** et l'erreur standard décroît en `1/√K` : pente **−1/2** en échelle log-log.
- Dans chaque permutation, les contributions somment à `v(N)` ; l'efficacité est donc exactement conservée.
- L'erreur standard est calculée **séparément pour chaque joueur**.

### 3.3 Cœur

Le cœur est l'ensemble des allocations stables : aucune coalition n'a intérêt à faire sécession.

```
Cœur(v) = { x ∈ ℝⁿ : Σ_i x_i = v(N)   et   Σ_{i∈S} x_i ≥ v(S)  pour tout S ⊆ N, S ≠ ∅ }
```

C'est un système linéaire (un polyèdre). Avec `scipy.optimize.linprog` :

- on teste d'abord sa **faisabilité** ; s'il est infaisable, le cœur est **vide** et la fonction renvoie `None` ;
- sinon, pour chaque joueur `i`, on résout `min x_i` puis `max x_i` sous les mêmes contraintes : **2n programmes linéaires**.

Écriture pour `linprog` (qui minimise sous `A_ub x ≤ b_ub`) : la contrainte `Σ_{i∈S} x_i ≥ v(S)` devient `−Σ_{i∈S} x_i ≤ −v(S)`, et l'efficacité va dans `A_eq, b_eq`. Pour maximiser `x_i`, on minimise `−x_i`. Les variables doivent être **non bornées** (`bounds=(None, None)`), car `linprog` impose `x ≥ 0` par défaut.

### 3.4 Indice de Shapley-Shubik

Dans un jeu simple, le joueur `i` est **pivot** d'un ordre si son arrivée fait passer la coalition de ses prédécesseurs `S` de perdante à gagnante : `v(S) = 0` et `v(S ∪ {i}) = 1`.

```
φ_SS,i = (nombre d'ordres où i est pivot) / n!
       = Σ_{S ⊆ N\{i}, v(S)=0, v(S∪{i})=1}  |S|!(n−|S|−1)!/n!
```

C'est exactement la valeur de Shapley du jeu simple. Pour un vote pondéré `[q ; w_1, …, w_n]`, `v(S) = 1` si `Σ_{i∈S} w_i ≥ q`, sinon `0`.

---

## 4. Pseudo-codes

### Algorithme 1 – Valeur de Shapley exacte par coalitions (masques de bits)

```
ENTRÉES : v (fonction caractéristique sur masques), n (nombre de joueurs)
SORTIE  : φ[0..n-1]

1.  Précalculer fact[k] = k! pour k = 0..n
2.  φ[i] ← 0 pour i = 0..n-1
3.  POUR S de 0 à 2ⁿ − 1 FAIRE                       // toutes les coalitions
4.      k ← nombre de bits à 1 dans S                // |S|
5.      c ← fact[k] · fact[n − k − 1] / fact[n]      // poids (valable car i ∉ S ⇒ k ≤ n−1)
6.      POUR i de 0 à n−1 FAIRE
7.          SI i ∉ S ALORS                           // ((S >> i) & 1) == 0
8.              φ[i] ← φ[i] + c · ( v(S | (1 << i)) − v(S) )
9.  RETOURNER φ
```

Complexité : `O(n·2ⁿ)` appels à `v`. Contrôle : `Σ φ = v(N)`.

### Algorithme 2 – Valeur de Shapley par Monte-Carlo

```
ENTRÉES : v, n, K (K ≥ 2), seed (optionnel)
SORTIE  : (φ̂[0..n-1], SE[0..n-1])

1.  SI seed ≠ None ALORS initialiser le générateur aléatoire avec seed
2.  somme[i] ← 0 ; somme_carres[i] ← 0   pour i = 0..n-1
3.  POUR k de 1 à K FAIRE
4.      π ← permutation aléatoire uniforme de {0, …, n-1}
5.      S ← 0 ; v_prec ← v(0)
6.      POUR j de 0 à n−1 FAIRE
7.          i ← π[j]
8.          S ← S | (1 << i)
9.          v_cour ← v(S)
10.         m ← v_cour − v_prec                      // contribution marginale de i
11.         somme[i] ← somme[i] + m
12.         somme_carres[i] ← somme_carres[i] + m²
13.         v_prec ← v_cour
14. POUR i de 0 à n−1 FAIRE
15.     φ̂[i] ← somme[i] / K
16.     s² ← ( somme_carres[i] − K · φ̂[i]² ) / (K − 1)   // variance empirique non biaisée
17.     SE[i] ← √( max(s², 0) / K )
18. RETOURNER (φ̂, SE)
```

Complexité : `O(K·n)` appels à `v`. Le `max(s², 0)` protège des erreurs d'arrondi.

### Algorithme 3 – Cœur par programmation linéaire

```
ENTRÉES : v, n
SORTIE  : [(min_0, max_0), …, (min_{n-1}, max_{n-1})]  ou  None si le cœur est vide

1.  Construire le système, variables x_0..x_{n-1} non bornées :
        égalité     : Σ_i x_i = v(N)                         // efficacité
        inégalités  : −Σ_{i∈S} x_i ≤ −v(S)                   // pour tout S, S ≠ ∅, S ≠ N
2.  Résoudre le LP de faisabilité (objectif nul) avec linprog
3.  SI le LP est infaisable ALORS RETOURNER None             // cœur vide
4.  POUR i de 0 à n−1 FAIRE
5.      min_i ← valeur optimale de   min  x_i   sous les contraintes de l'étape 1
6.      max_i ← − valeur optimale de min −x_i   sous les contraintes de l'étape 1
7.  RETOURNER [(min_i, max_i) pour i = 0..n-1]
```

Coût : `1 + 2n` LP (le premier teste le cœur vide ; les `2n` suivants donnent les bornes). Le cas `S = N` est couvert par l'égalité d'efficacité.

### Algorithme 4 – Indice de Shapley-Shubik par joueurs pivots

```
ENTRÉES : v (jeu simple : v(S) ∈ {0,1}, v(0) = 0, v(N) = 1), n
SORTIE  : φ_SS[0..n-1]

1.  Précalculer fact[k] = k! pour k = 0..n
2.  φ_SS[i] ← 0 pour i = 0..n-1
3.  POUR S de 0 à 2ⁿ − 1 FAIRE
4.      SI v(S) = 0 ALORS                                // S est perdante
5.          k ← |S|
6.          c ← fact[k] · fact[n − k − 1] / fact[n]
7.          POUR i de 0 à n−1 FAIRE
8.              SI i ∉ S ET v(S | (1 << i)) = 1 ALORS    // i est pivot après S
9.                  φ_SS[i] ← φ_SS[i] + c
10. RETOURNER φ_SS
```

Équivalent « par ordres » (définition) : pour chaque permutation `π`, trouver le premier joueur dont l'arrivée rend la coalition gagnante, lui ajouter `1`, puis diviser tous les compteurs par `n!`. Coût `O(n·n!)` contre `O(n·2ⁿ)` ci-dessus. Contrôle : `Σ φ_SS = 1`.

---

## 5. Exemples de vérification

### 5.1 Jeu à 3 joueurs du cours (joueurs 1, 2, 3 ↔ indices 0, 1, 2)

`v({1,2}) = v({1,3}) = 60`, `v({2,3}) = 0`, `v(N) = 90`, singletons supposés nuls.

- **Shapley** : `φ = (50, 20, 20)`. Pour le joueur 1 : `0 + 1/6·60 + 1/6·60 + 1/3·90 = 50`. Somme `= 90 = v(N)` ✓.
- **Cœur** : bornes `x_1 ∈ [30, 90]`, `x_2 ∈ [0, 30]`, `x_3 ∈ [0, 30]`.
  - `(50, 20, 20)` est dans le cœur.
  - `(0, 45, 45)` est **hors du cœur** : `x_1 + x_2 = 45 < 60 = v({1,2})`.

### 5.2 Vote pondéré `[51 ; 40, 30, 20, 10]`

| Joueur | Poids | Indice de Shapley-Shubik |
|---|---|---|
| 1 | 40 | 5/12 ≈ 0,417 |
| 2 | 30 | 1/4 = 0,250 |
| 3 | 20 | 1/4 = 0,250 |
| 4 | 10 | 1/12 ≈ 0,083 |

Somme `= 1` ✓. Le joueur 4 (poids 10) n'est pivot que lorsque 2 et 3 le précèdent seuls.

---

## 6. Limites et erreurs fréquentes

- **Oublier `v(∅) = 0`** : l'efficacité n'est alors plus garantie.
- **Mélanger jeu de gains et jeu de coûts** : le cœur d'un jeu de coûts s'écrit avec `≤` ; réutiliser les inégalités d'un jeu de gains donne un résultat faux.
- **Croire que le cœur est toujours non vide** : il peut être vide (`coeur_lp` renvoie alors `None`), et la valeur de Shapley n'est pas toujours dans le cœur.
- **Explosion combinatoire** : `2ⁿ` coalitions (et `2ⁿ − 2` contraintes pour le cœur) limitent le calcul exact à `n ≈ 20` ; au-delà, Monte-Carlo.
- **Monte-Carlo** : l'estimation est aléatoire ; fixer `seed` pour reproduire un résultat, et choisir `K` selon l'erreur standard visée.
- **Shapley-Shubik hors jeu simple** : l'indice n'a de sens que pour `v ∈ {0, 1}` monotone.
- **Clés JSON** : des clés `frozenset` ne sont pas compatibles avec le JSON de l'outil ; utiliser des chaînes de type `"1,2"`.
