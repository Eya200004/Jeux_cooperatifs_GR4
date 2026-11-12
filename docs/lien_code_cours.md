# Lien avec le code du cours

Le code du cours a déjà un outil valeur_shapley, mais il est lent
quand il y a beaucoup de joueurs.

Notre projet ajoute à l'agent :
1. Une version rapide de valeur_shapley (Monte-Carlo) avec une limite
   sur le nombre de joueurs.
2. Un outil coeur(joueurs, v) : dit si un partage est stable.
3. Un outil shapley_vote(quota, poids) : calcule le poids de chacun
   dans un vote.

Les coalitions sont écrites en JSON avec des clés comme "1,2".
Les outils reçoivent des données, jamais du code.