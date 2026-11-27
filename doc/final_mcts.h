#ifndef FINAL_MCTS_H
#define FINAL_MCTS_H

// =============================================================================
//  FINAL_MCTS — Version épurée et commentée du joueur « MCTS ResNet » de l'app
// =============================================================================
//
//  À QUOI SERT CE FICHIER
//  ----------------------
//  Ce fichier (et son compagnon doc/final_mcts.cpp) contient le code du
//  joueur « MCTS ResNet » de l'application Yolah, dans sa configuration de
//  l'application (config/hybrid/resnet.cfg, identique aux réglages de l'app) :
//  une recherche arborescente Monte-Carlo à la AlphaZero, avec les
//  raffinements de KataGo, guidée par un réseau de neurones résiduel (ResNet)
//  à deux têtes (valeur + politique) qui tourne sur la carte graphique.
//
//  Le vrai code est réparti dans plusieurs fichiers :
//    player/alphazero_mcts.{h,cpp}        la recherche (az::Search)
//    player/alphazero_mcts_player.{h,cpp} le joueur (lecture du JSON)
//    player/nn_cache.h                    le cache des sorties du réseau
//    nnue/nn_evaluator.h                  l'interface recherche ↔ réseau, l'encodage
//    nnue/resnet_torch.{h,cpp}            le réseau exécuté par libtorch (GPU)
//    nnue/batched_evaluator.{h,cpp}       le regroupement des requêtes des threads
//    nnue/cnn_resnet_value_policy.py      l'architecture et l'entraînement (Python)
//    nnue/resnet_export.py                l'export du réseau vers le C++
//  Ils contiennent en plus tout ce qui sert à l'auto-apprentissage (bruit de
//  Dirichlet, tirage des coups avec température, « forced playouts »,
//  « playout cap randomization »), un mode arbre (sans graphe), un backend CPU
//  (Eigen) et des options jamais utilisées par l'app. Tout cela est retiré ici,
//  et ces fichiers-ci se lisent seuls.
//
//  « Même code » veut dire : compilé à côté du vrai joueur, avec un seul thread
//  de recherche et un nombre fixe de simulations, ce code donne exactement les
//  mêmes visites, les mêmes valeurs et le même coup sur les positions de test
//  (voir « VÉRIFICATION » à la fin de ce commentaire). Avec plusieurs threads,
//  aucune des deux versions n'est reproductible d'une exécution à l'autre (les
//  threads se partagent l'arbre dans un ordre qui dépend du système), mais le
//  code exécuté est le même.
//
//  ---------------------------------------------------------------------------
//  LE JEU EN BREF
//  ---------------------------------------------------------------------------
//  (Les règles en détail sont au début de doc/final_dev.h.) Plateau 8 × 8,
//  4 pièces par camp, Noir commence. Une pièce se déplace comme une dame
//  d'échecs vers une case libre, sans rien sauter ; sa case de départ devient
//  un TROU définitif. Chaque coup rapporte 1 point ; un camp bloqué passe
//  (`Move::none()`, la valeur 0 = « a1 → a1 ») ; la partie finit quand les deux
//  camps sont bloqués, et gagne celui qui a le plus de points. Conséquence
//  essentielle pour la recherche : le nombre de trous augmente de 1 à chaque
//  coup (hors passe), donc une position ne se répète JAMAIS (graphe sans
//  cycle), alors que la même position est atteinte par de très nombreux ordres
//  de coups (les 4 pièces d'un camp bougent indépendamment). Une case = un
//  entier 0..63 (a1 = 0, b1 = 1, …, h8 = 63) ; un ensemble de cases = un
//  `uint64_t` (« bitboard », bit s = case s) ; le pli `ply` compte les
//  demi-coups depuis le début de la partie ; le camp au trait est `ply & 1`.
//
//  ---------------------------------------------------------------------------
//  L'IDÉE DE LA RECHERCHE (MCTS « à la AlphaZero »)
//  ---------------------------------------------------------------------------
//  Le minimax (doc/final_dev.h) explore tous les coups jusqu'à une profondeur
//  fixe et évalue les feuilles avec un petit réseau rapide (NNUE). Le MCTS
//  fait l'inverse : il construit un arbre très déséquilibré, profond sur les
//  coups prometteurs et presque vide ailleurs, et chaque position nouvelle est
//  évaluée par un GROS réseau (environ 36 millions de paramètres, 1000 fois
//  plus de calcul par position que le NNUE) qui donne deux choses :
//    • une VALEUR v ∈ (−1, 1) : le résultat espéré pour le camp au trait
//      (+1 gagné, −1 perdu, 0 nul) ;
//    • une POLITIQUE : une probabilité a priori P(s, a) pour chaque coup a,
//      « la chance que ce soit le meilleur coup », qui dit où chercher.
//  Une SIMULATION (ou « visite ») fait :
//    1. SÉLECTION : on descend de la racine en choisissant à chaque nœud le
//       coup qui maximise PUCT = Q + U (exploitation + exploration, section 3
//       du .cpp), jusqu'à une position jamais évaluée (une FEUILLE) ;
//    2. ÉVALUATION : le réseau donne v et les P de la feuille ; elle est
//       « développée » (ses coups deviennent des arêtes, avec leurs P) ;
//    3. REMONTÉE (« backup ») : v est ajouté à la somme des valeurs de chaque
//       nœud du chemin, avec un changement de signe à chaque pli (ce qui est
//       bon pour l'un est mauvais pour l'autre).
//  Q(s, a) est la moyenne des valeurs remontées par le coup a : plus un coup a
//  été visité, plus son Q est fiable. Le coup joué à la fin est en gros le
//  plus visité (section 7 du .cpp : la borne inférieure de confiance, LCB).
//  Il n'y a pas de « partie aléatoire jusqu'au bout » comme dans le MCTS
//  classique : la valeur du réseau la remplace (seules les fins de partie
//  réellement atteintes dans l'arbre donnent un résultat exact).
//
//  Dans l'app, la recherche dure le temps donné par le niveau (1 s par coup
//  dans les campagnes), avec 16 threads qui envoient chacun 16 feuilles à la
//  fois au réseau. Ordres de grandeur mesurés (RTX 3080, fp16) : le réseau
//  évalue ~7 500 à 10 000 positions par seconde par lots de 256 ; la recherche
//  fait ~2 500 à 2 900 nouvelles feuilles par seconde en milieu de partie
//  (mesure du 23/11/2026, campagne H2), donc ~2 500 évaluations par coup à
//  1 s — 1000 fois moins de positions que le minimax dev dans le même temps.
//
//  ---------------------------------------------------------------------------
//  PETIT GLOSSAIRE
//  ---------------------------------------------------------------------------
//  Toutes les notations s'appuient sur ce morceau de graphe de recherche
//  (les nombres sont des visites ; A, B, C sont des coups de Noir, X un coup
//  de Blanc, et A, B déplacent deux pièces DIFFÉRENTES de Noir) :
//
//                           r              Noir au trait (racine)
//                       A ╱   ╲ B
//                       ╱       ╲
//                     ▼           ▼
//                     p           q          Blanc au trait
//                   X │           │ X
//                     ▼           ▼
//                     s           t          Noir au trait
//                  C ╱ ╲ B       ╱ A
//                 6 ╱   ╲ 10    ╱ 30
//                  ▼     ╲     ╱
//                  d      ▼   ▼
//                           c                Blanc au trait
//
//      Les nombres sur les flèches sont les visites passées par chaque arête.
//      N(c) = 40 (10 venues de s + 30 venues de t), N(d) = 6.
//
//    • Nœud (`Node`) : une position (r, p, q, s, t, c, d). Arête (`Edge`) : un
//      coup joué depuis une position (s ──B──► c), avec son a priori P(s, B)
//      donné par le réseau et un pointeur vers le nœud d'arrivée.
//    • GRAPHE, pas arbre : « A, X, B » et « B, X, A » donnent la même
//      position, donc un seul nœud c, qui a deux parents (s et t). Tous les
//      chemins qui mènent à c partagent ce qu'on sait de c (section 2 du .cpp).
//      Il n'y a jamais de cycle : chaque coup ajoute un trou.
//    • N(c) : visites TERMINÉES du nœud c, par n'importe quel parent
//      (`Node::n`). Ici N(c) = 10 + 30 = 40.
//      N(s, B) : visites terminées passées par l'ARÊTE s ──B──► c, la part de
//      CE parent (`Edge::n`). Ici N(s, B) = 10 et N(t, A) = 30.
//      Dans un arbre, N(s, B) et N(c) seraient égaux ; dans le graphe, N(c)
//      cumule les N des arêtes qui y entrent.
//    • W(c) : somme des valeurs remontées dans c, du point de vue du camp qui
//      a JOUÉ le coup menant à c (Noir, qui joue B ou A), donc de l'adversaire
//      du camp au trait dans c. Q(c) = W(c) / N(c) : la valeur moyenne, même
//      point de vue. Ainsi, en s, Noir compare directement Q(c) et Q(d) : il
//      veut le plus grand. Et Q(s, B), « ce que vaut le coup B depuis s »,
//      est simplement Q(c).
//      Une visite remonte sa valeur dans TOUS les nœuds de son chemin, avec
//      un changement de signe à chaque pli : une visite r → q → t → c ajoute
//      à W(c), W(t), W(q) et W(r), mais pas à W(s). W(s) n'est donc fait que
//      des visites passées par s : la première, qui a évalué s quand c'était
//      une feuille, puis les 10 vers c et les 6 vers d (N(s) = 17).
//    • W2 : somme des carrés des valeurs remontées (pour l'écart-type, σ̂ en
//      section 3).
//    • PUCT (« Predictor + Upper Confidence bounds applied to Trees ») : la
//      formule de choix du coup à chaque nœud de la descente (section 3) ;
//      en s :
//          score(B) = Q(c) + E · P(s, B) / (1 + N(s, B))
//                     ────   ──────────────────────────
//                     valeur  bonus d'exploration
//      Les deux termes ne comptent pas les mêmes visites :
//      – la VALEUR prend Q(c), la moyenne des 40 visites de c : c est la même
//        position par les deux chemins, donc les 30 visites venues de t la
//        renseignent tout autant que les 10 venues de s ;
//      – le BONUS divise par 1 + N(s, B) = 11, et non par 1 + N(c) = 41.
//        Explication en quatre temps.
//
//        a) Ce que fait le bonus. Q seul ne suffit pas : depuis s, on
//           enverrait toutes les visites vers le coup dont le Q est le plus
//           grand à cet instant, et un coup mal jugé au début ne serait plus
//           jamais réessayé. Le bonus fait PARTAGER les visites de s entre B
//           et C, selon ce mécanisme de bascule :
//             • chaque visite de s vers B ajoute 1 à N(s, B) : le bonus de B
//               baisse ;
//             • dans le même temps, E (qui contient √W_tot, avec W_tot =
//               N(s, B) + N(s, C), le total des visites de s) augmente : le
//               bonus de C, qui n'a pas été visité, monte ;
//             • tôt ou tard C repasse devant, reçoit des visites, et ainsi de
//               suite. Au total, chaque coup reçoit une part des visites de s,
//               d'autant plus grande que son Q et son a priori P sont grands.
//
//        b) Ce que le bonus doit compter. C'est un compteur de PARTAGE : il
//           doit compter les visites que s a déjà données à B, c'est-à-dire
//           N(s, B) = 10. Les 30 visites venues de t ne sont pas des visites
//           de s : elles n'ont rien pris sur la part de B. Elles ne sont pas
//           perdues pour autant : elles sont dans Q(c), qui est la moyenne des
//           40 visites de c (premier terme de la formule).
//           Exemple chiffré, avec W_tot = 16 visites de s (10 vers B, 6 vers
//           C), E ≈ √16 = 4 et P = 0.5 pour les deux coups :
//               bonus de B = 4 · 0.5 / (1 + 10) = 0.18
//               bonus de C = 4 · 0.5 / (1 +  6) = 0.29
//           En divisant par 1 + N(c), le bonus de B tomberait à
//           4 · 0.5 / (1 + 40) = 0.05, presque quatre fois moins.
//
//        c) Ce qui se passerait en comptant N(c). L'idée n'est pas absurde :
//           dans la formule, plus le dénominateur est grand, moins l'a priori
//           P compte et plus on se fie à Q. Or Q(c) repose sur 40 visites :
//           il est fiable, et se fier à lui plutôt qu'à P(s, B) paraît juste.
//           Le problème est ailleurs : la formule ne juge pas B seul, elle
//           COMPARE B et C, et les deux ne seraient plus traités de la même
//           façon. Le bonus de B serait divisé par 41 (visites de s ET de t),
//           celui de C par 1 + N(s, C) (d n'a que les visites de s). B
//           n'aurait presque plus de bonus, C garderait le sien, qui de plus
//           monte avec W_tot à chaque visite de s vers B. Dès que Q(c) et
//           Q(d) sont proches, C l'emporterait presque toujours.
//           On peut objecter qu'explorer C n'est pas inutile : d est moins
//           bien connu que c. Mais l'effet va bien au-delà : C finirait par
//           recevoir la MAJORITÉ des visites de s alors que B est meilleur.
//           Les visites de s ne diraient plus « quel coup est le meilleur »
//           mais « quel coup est le moins connu ». Or, comme le montre la
//           partie d), on se sert justement des visites de s pour désigner son
//           meilleur coup.
//           Modèle jouet : Q(c) = 0.30 et Q(d) = 0.25 fixes, P = 0.5 chacun,
//           100 visites de s ; on compte combien vont vers B et vers C :
//               bonus divisé par   visites de c venues de t    B / C
//               1 + N(s, B)          (sans effet)              62 / 38
//               1 + N(c)                  30                   55 / 45
//               1 + N(c)                 100                   42 / 58
//               1 + N(c)                 300                   24 / 76
//           Avec la vraie formule (première ligne), B, le meilleur, reçoit le
//           plus de visites, quel que soit le nombre de visites venues de t.
//           En comptant N(c), à partir d'environ 100 visites venues de t, C en
//           reçoit davantage. Si Q(d) = 0.28 au lieu de 0.25, avec 300 visites
//           venues de t, B ne reçoit plus AUCUNE visite de s (0 / 100).
//
//        d) Pourquoi ce serait grave. Le partage N(s, B) / N(s, C) n'est pas
//           un détail interne, il sert à trois choses :
//             1. si s est la racine, le coup JOUÉ est le plus visité
//                (section 7 du .cpp) : on jouerait C ;
//             2. en self-play (parties jouées pour entraîner le réseau), quand
//                s est la racine, la politique que le réseau apprend pour s
//                est ce partage, N(s, B) / N(s) et N(s, C) / N(s) : il
//                apprendrait à préférer C ;
//             3. la valeur de s est la moyenne de ce qui remonte par s : une
//                visite vers B rapporte à peu près Q(c), une vers C à peu
//                près Q(d), donc, pour Noir,
//                    valeur de s ≈ (N(s, B) · Q(c) + N(s, C) · Q(d)) / N(s).
//                Trop de visites vers C la tire vers Q(d) : s paraît moins
//                bon pour Noir qu'il ne l'est, et p (Blanc) juge mal le coup X
//                qui mène à s.
//           Avec N(s, B), s partage ses visites entre B et C exactement comme
//           dans un arbre ; le graphe lui apporte seulement un Q(c) plus
//           précis, grâce aux visites venues de t.
//    • FPU (« First Play Urgency ») : la valeur supposée d'un coup jamais
//      essayé, à la place de Q(c) quand c n'a encore aucune visite : à peu près
//      la valeur du parent, moins une petite pénalité (section 3).
//    • Perte virtuelle (« virtual loss ») : une descente en cours (pas encore
//      remontée) compte provisoirement comme une défaite dans les nœuds
//      qu'elle traverse, et comme une visite dans N(s, B), pour que les autres
//      descentes (autres threads, ou la suite du même lot) aillent ailleurs
//      (section 5).
//    • Lot (« batch ») : plusieurs feuilles envoyées ensemble au réseau. Un GPU
//      ne va vite que sur des lots : évaluer 256 positions coûte à peine plus
//      cher que d'en évaluer une.
//    • Elo : l'échelle des écarts de force utilisée dans les résultats de
//      matchs cités ici (« +4 Elo », etc.). Un écart de d Elo entre deux
//      joueurs correspond au score moyen attendu du plus fort (victoire = 1
//      point, nul = ½, défaite = 0) :
//          score = 1 / (1 + 10^(−d / 400))
//      soit, pour le plus fort : +10 Elo ≈ 51,4 % des points, +100 ≈ 64 %,
//      +400 ≈ 91 %. Inversement, un match gagné 55 % des points mesure un
//      écart d'environ +35 Elo.
//
//  ---------------------------------------------------------------------------
//  ÉCHELLE DES VALEURS
//  ---------------------------------------------------------------------------
//    • Toutes les valeurs de la recherche sont dans [−1, 1], du point de vue
//      indiqué à chaque fois (camp au trait, ou camp qui vient de jouer).
//    • Le réseau donne tanh(…) ∈ (−1, 1), entraîné sur le résultat z ∈
//      {−1, 0, +1} : v ≈ E[z], et P(gain) ≈ (v + 1) / 2 si les nuls sont rares.
//    • Une fin de partie réellement atteinte vaut (« utilité de score », à la
//      KataGo, u = 0.05, échelle 10) :
//          (1 − u)·signe(d) + u·tanh(d / 10),     d = écart de points final
//      pour le camp au trait : une victoire vaut entre 0.95 et 1, une victoire
//      large plus qu'une courte, et une victoire bat toujours un nul (0). Le
//      réseau, lui, ne prédit pas de score : seules les fins réelles en ont
//      (campagne de l'utilité de score : +4 Elo, des victoires plus nettes ;
//      sans elle, une fois la partie gagnée, tous les coups gagnants se
//      valent et la recherche joue n'importe lequel).
//
//  ---------------------------------------------------------------------------
//  LE RÉSEAU (ResNet 256 × 30, nnue/cnn_resnet_value_policy.py, classe Net)
//  ---------------------------------------------------------------------------
//  ENTRÉE : 4 plans de 8 × 8 nombres (0 ou 1), soit un tenseur (4, 8, 8) :
//      plan 0 : pièces noires,  plan 1 : pièces blanches,
//      plan 2 : trous (cases vidées, « empty » dans le code),
//      plan 3 : le camp au trait (tout à 0 = Noir, tout à 1 = Blanc).
//  Les points déjà marqués ne sont pas donnés : ils se déduisent du nombre de
//  trous de chaque camp... à peu près (une passe ne fait pas de trou, et le
//  réseau ne sait pas de quel camp vient un trou). L'ordre des cases dans un
//  plan est particulier (encode_planes dans le .cpp, section 9) : les colonnes
//  y sont en miroir. C'est sans importance pour le réseau, mais l'entraînement
//  et le C++ doivent faire exactement la même chose.
//
//  TRONC (les « 256 × 30 » : 256 canaux, 30 blocs résiduels) :
//
//      (4,8,8) ─► conv 3×3, 4→256 ─► BN ─► ReLU                    « stem »
//              ─► 30 × bloc résiduel                                (ci-dessous)
//              ─► BN ─► ReLU ─► carte de caractéristiques (256, 8, 8)
//
//    Bloc résiduel « pré-activation » (ResNet v2) :
//                          ┌─────────────── identité ───────────────┐
//      x ─► BN ─► ReLU ─► conv 3×3 ─► BN ─► ReLU ─► conv 3×3 ─────► (+) ─► sortie
//    Les convolutions 3×3 ont un « padding » de 1 : la carte reste 8 × 8 tout
//    du long, chaque case voit ses 8 voisines à chaque couche ; après 60
//    convolutions, toute case « voit » tout le plateau (il suffit de 7 pour
//    aller d'un coin à l'autre). L'identité (le saut) permet d'entraîner un
//    réseau aussi profond : le gradient passe directement d'un bloc à l'autre.
//    BN = BatchNorm : normalise chaque canal (moyenne 0, variance 1 sur le lot
//    pendant l'entraînement ; des constantes apprises ensuite). ReLU(x) =
//    max(0, x).
//
//  TÊTE DE VALEUR (AlphaZero) :
//      (256,8,8) ─► conv 1×1, 256→1 ─► BN ─► ReLU ─► aplatir (64)
//                ─► linéaire 64→256 ─► ReLU ─► linéaire 256→1 ─► tanh ─► v
//  TÊTE DE POLITIQUE (AlphaZero) :
//      (256,8,8) ─► conv 1×1, 256→2 ─► BN ─► ReLU ─► aplatir (128)
//                ─► linéaire 128→4096 ─► 4096 « logits »
//    Un logit par paire (départ, arrivée) : indice = départ × 64 + arrivée,
//    départ et arrivée en 0..63 (a1 = 0). Les trois quarts sont des coups
//    impossibles : on ne les masque pas à l'entraînement (ils ne sont jamais la
//    cible, leur probabilité tombe toute seule) ; la recherche ne lit que les
//    logits des coups légaux et fait le softmax sur eux seuls (section 4 du
//    .cpp). La passe (a1 → a1) a l'indice 0, qui n'est jamais un vrai coup.
//
//  TAILLE ET COÛT :
//    • stem 4·256·9 = 9 216 poids ; chaque conv 3×3 du tronc 256·256·9 =
//      589 824 poids, 60 convolutions : 35,4 millions ; tête de politique
//      128·4096 + 4096 = 528 384 ; tête de valeur ~16 900 ; BN ~31 000.
//      Total ≈ 35,9 millions de paramètres (≈ 72 Mo en fp16).
//    • Calcul par position : chaque conv 3×3 fait 589 824 multiplications-
//      additions par case, × 64 cases = 37,7 millions ; × 60 ≈ 2,3 milliards
//      de multiplications-additions par position (~4,5 GFLOP). C'est pourquoi
//      le réseau tourne sur le GPU, en demi-précision (fp16), par lots.
//
//  ENTRAÎNEMENT (supervisé, nnue/cnn_resnet_value_policy*.py) :
//    • Données : les positions des parties des fichiers games_* (écrits par
//      data/generate_games.cpp : quelques coups aléatoires au début, puis deux
//      joueurs programmés ; les positions des coups aléatoires et les parties
//      entièrement aléatoires sont exclues), encodées une fois pour toutes par
//      nnue/preprocess.py.
//    • Cibles : pour la valeur, le résultat final z ∈ {−1, 0, +1} du point de
//      vue du camp au trait ; pour la politique, le coup réellement joué dans
//      la partie (indice départ × 64 + arrivée).
//    • Perte : MSE(v, z) + entropie croisée(logits, coup joué), poids 1 et 1
//      (comme AlphaZero). Optimiseur Adam, pas 0.001 qui décroît en cosinus
//      jusqu'à 1e-5, « weight decay » 1e-4 (version par blocs,
//      cnn_resnet_value_policy_chunked.py, celle que lance le .sh).
//      Entraînement sur plusieurs GPU (DDP), en précision mixte (bf16 sur les
//      GPU Ampere et plus récents, fp16 sinon).
//
//  EXPORT VERS LE C++ (nnue/resnet_export.py) :
//    • <nom>.ts : le réseau en TorchScript, « tracé » en mode évaluation
//      (les BatchNorm deviennent des constantes). C'est ce fichier que charge
//      l'app (models/ ou nnue/cnn_resnet_256x30_value_policy.ts) et que lit
//      la classe TorchNet du .cpp (section 9). Sortie : un tuple (valeurs (n,),
//      logits (n, 4096)) pour une entrée (n, 4, 8, 8).
//    • <nom>.bin : les poids à plat, BN repliées dans les convolutions, pour un
//      backend C++ pur (Eigen) sans GPU, très lent (~150 positions/s), retiré ici.
//    • <nom>.test.bin : quelques positions avec les sorties exactes de PyTorch,
//      pour vérifier les backends C++ (test/alphazero_check_main.cpp).
//
//  ---------------------------------------------------------------------------
//  L'INTERFACE AVEC LE C++ (sections 8 à 10 du .cpp)
//  ---------------------------------------------------------------------------
//  La recherche ne manipule jamais de tenseur. Elle donne une liste de
//  REQUÊTES (une position + ses coups légaux) et reçoit pour chacune un
//  RÉSULTAT (la valeur, et les logits de ces coups-là, dans le même ordre) :
//
//     threads de recherche              regroupeur (Batcher)        réseau (TorchNet)
//     ───────────────────              ────────────────────        ─────────────────
//     thread 0 : 16 feuilles ──┐
//     thread 1 : 16 feuilles ──┼──►  file d'attente ──► un lot de ≤ 256 ──► GPU, une passe
//     …                        │     (attend que tous les                  (n,4,8,8) → v, logits
//     thread 15: 16 feuilles ──┘      threads soient là,       ◄── résultats redistribués
//                                     ou 0,5 ms au plus)            à chaque thread
//
//    • TorchNet (copie commentée de nnue/resnet_torch.cpp) : encode les
//      positions dans un tampon CPU « épinglé » (copie DMA rapide vers le GPU),
//      le copie sur le GPU en fp16, fait UNE passe avant, recopie les deux
//      sorties, et extrait pour chaque position les logits de ses coups légaux.
//    • Batcher (copie de nnue/batched_evaluator.cpp) : 16 threads × 16
//      feuilles = des lots de 256 au lieu de 16. Sans lui, chaque thread
//      appellerait le GPU avec 16 positions : 16 fois plus d'appels.
//    • NNCache (copie de player/nn_cache.h) : la sortie du réseau ne dépend que
//      du plateau et du camp au trait ; ces paires reviennent souvent par
//      transposition (et d'un coup à l'autre), on les garde en mémoire (64 Mo).
//
//  ---------------------------------------------------------------------------
//  CONFIGURATION DE L'APP (config/hybrid/resnet.cfg) ET VALEURS PAR DÉFAUT
//  ---------------------------------------------------------------------------
//    microseconds 1 000 000 (selon le niveau)   nb threads 16   batch size 16
//    cpuct exploration 1.0, log 0.45, base 500  (section 3)
//    cpuct utility stdev scale 0.85, prior 0.40, prior weight 2.0
//    fpu reduction 0.2, root fpu reduction 0.1 (pondération par la masse
//      d'a priori visitée : oui, puissance 1 ; « loss prop » 0)
//    policy temperature 1.0   graph search oui   reuse tree oui   nn cache 64 Mo
//    use lcb oui, lcb stdevs 5, min visit prop for lcb 0.2
//    policy target pruning oui (il modifie aussi le choix du coup, section 7)
//    score utility 0.05, score scale 10
//    backend torch, device cuda, fp16 oui ; « merged batch size » = 16 × 16
//  Ces valeurs sont celles de KataGo (cpp/search/searchparams.cpp), sauf
//  l'utilité de score (spécifique à Yolah). Elles n'ont pas été réglées pour
//  Yolah.
//
//  ---------------------------------------------------------------------------
//  VÉRIFICATION
//  ---------------------------------------------------------------------------
//  test/final_mcts_check.cpp (hors de la compilation normale ; compilé par
//  test/final_mcts_check.sh, lancé depuis config/) joue les deux versions côte à côte sur les positions
//  de test/bench_positions.txt : un thread, 800 simulations, réseau en fp32,
//  et compare pour chaque position le coup, les visites et le Q de chaque
//  coup de la racine. Toute modification de ce code doit être revérifiée.
//
//  Dernière vérification (23/11/2026, RTX 3080) : 209 / 209 positions
//  identiques, 411 recherches (la position, puis deux plis plus loin depuis
//  le graphe réutilisé) : mêmes coups, visites, Q et valeurs de jeu.
// =============================================================================

#include "game.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <torch/script.h>

namespace final_mcts {

// =============================================================================
//  LES DÉCLARATIONS
// =============================================================================
//  Vue d'ensemble de qui appelle qui pendant un coup de l'app :
//
//      Player::play(position)
//        └─► Search::search(position)                     sections 1 à 7 du .cpp
//              ├─ racine : reprise du graphe précédent si possible ; sinon
//              │  neuve, évaluée SEULE et tout de suite (1 appel au réseau)
//              ├─ 16 threads run_worker, chacun répète jusqu'à la fin du temps :
//              │    1. RASSEMBLER : jusqu'à 16 descentes depuis la racine (PUCT).
//              │       Chaque descente qui arrive sur un nœud jamais évalué :
//              │         • fin de partie → valeur exacte remontée AUSSITÔT ;
//              │         • position dans le NNCache → nœud développé et valeur
//              │           remontée AUSSITÔT, sans réseau ;
//              │         • sinon → la feuille est mise dans le lot du thread et
//              │           marquée EXPANDING ; elle n'est PAS évaluée tout de
//              │           suite, sa descente attend avec elle (perte virtuelle
//              │           posée sur son chemin, pour que les suivantes aillent
//              │           ailleurs) ;
//              │         • feuille déjà dans le lot de CE thread (atteinte par
//              │           une autre descente) → la descente s'y joint : elle
//              │           remontera la même valeur, par son propre chemin ;
//              │         • feuille dans le lot d'un AUTRE thread → descente
//              │           annulée (une « collision »).
//              │       On arrête de rassembler après 16 visites (ou 16 collisions
//              │       avec les feuilles des autres threads).
//              │    2. ÉVALUER : UN appel Evaluator::evaluate pour tout le lot.
//              │       Le Batcher le fait encore attendre (au plus 0,5 ms, ou
//              │       jusqu'à ce que les 16 threads aient déposé le leur), puis
//              │       TorchNet évalue jusqu'à 16 × 16 = 256 positions en UNE
//              │       passe sur le GPU.
//              │    3. DÉVELOPPER chaque feuille (ses arêtes, ses a priori), la
//              │       ranger dans le NNCache, puis REMONTER la valeur de chaque
//              │       descente du lot par son propre chemin.
//              └─► make_result : la valeur de jeu de chaque coup (élagage, LCB)
//        ◄─ le coup choisi
//
//  Une feuille nouvelle attend donc deux fois avant d'avoir sa valeur : que
//  son thread ait fini de rassembler son lot, puis que le Batcher ait réuni
//  les lots des autres threads. C'est le prix des gros lots (sur le GPU,
//  évaluer 256 positions coûte à peine plus cher qu'en évaluer une) ; la
//  perte virtuelle limite le défaut : pendant l'attente, les autres descentes
//  évitent le chemin de cette feuille.
//
//  La recherche ne connaît le réseau qu'à travers l'interface Evaluator
//  (Request → Result) : elle ignore s'il y a un regroupeur, un GPU ou un CPU
//  derrière. C'est ce qui permet au Player de glisser le Batcher entre les
//  threads et le réseau, et à la vérification d'utiliser un seul thread sans.

// ── Requête et résultat : l'interface entre la recherche et le réseau ──────
// Une requête = une position à évaluer. Ses coups légaux sont fournis avec
// elle : la recherche les a déjà calculés (ils deviendront les arêtes), et
// le réseau n'a besoin de renvoyer que leurs logits, pas les 4096.
struct Request {
    const Yolah* state;     // la position à évaluer (pointe dans le tampon du thread, Worker::Leaf)
    const Move*  moves;     // ses coups légaux (dans l'ordre de Yolah::moves), au moins 1 (la passe s'il n'y a rien d'autre)
    uint16_t     nb_moves;  // leur nombre (≤ Yolah::MAX_NB_MOVES = 75)
};
// Le résultat d'une requête. Les logits sont bruts (avant softmax) : c'est la
// recherche qui fait le softmax sur les coups légaux (section 4 du .cpp).
// Taille fixe (75 logits), pour pouvoir préparer les tableaux de résultats une
// fois pour toutes, sans allocation pendant la recherche.
struct Result {
    float value;                        // valeur pour le camp au trait de la position, dans (−1, 1)
    float logits[Yolah::MAX_NB_MOVES];  // logits de politique des coups moves[0..nb_moves), même ordre
};

// Ce que la recherche sait d'un réseau : évaluer un lot. Deux réalisations :
//   • TorchNet : le réseau lui-même (un appel = une passe avant sur le GPU) ;
//   • Batcher : un intermédiaire qui regroupe les lots de plusieurs threads
//     en un seul, le passe à un TorchNet et redistribue les résultats.
// Les deux ont la même interface : un thread de recherche appelle evaluate()
// sans savoir lequel il a en face.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    // Évalue in.size() positions et écrit leurs résultats dans out[0..in.size()),
    // dans le même ordre ; out.size() >= in.size(). Appelable par plusieurs
    // threads à la fois. L'appel est BLOQUANT : il rend la main quand les
    // résultats sont écrits. Un `in` vide est permis (le Batcher s'en sert
    // comme d'un rendez-vous : attendre le prochain lot, section 10).
    virtual void evaluate(std::span<const Request> in, std::span<Result> out) = 0;
};

// ── 9. Le réseau, exécuté par libtorch (copie de nnue/resnet_torch.cpp) ────
// Charge le fichier .ts (un module TorchScript : le graphe de calcul et les
// poids, exécutable sans Python) et l'exécute par lots. Pour chaque lot :
// encode les positions en plans (4 × 8 × 8, avec le miroir des colonnes de
// l'entraînement), copie vers le GPU, une passe avant, copie des deux sorties
// (valeurs, 4096 logits) vers le CPU, et extraction des logits des coups
// légaux de chaque requête. Détails et raisons : section 9 du .cpp.
class TorchNet final : public Evaluator {
public:
    // filename : le .ts de resnet_export.py ; device : "cuda" (ou "cpu" ; sans
    // GPU disponible, "cuda" devient "cpu") ; fp16 : demi-précision (seulement
    // sur GPU) ; warm_batch : la plus grande taille de lot préparée au
    // chargement (256 = 16 threads × 16 feuilles). Le constructeur fait des
    // passes « à blanc » de tailles 1, 8, 32, 128 et warm_batch, pour que
    // cuDNN choisisse ses algorithmes avant le premier coup, pas pendant.
    TorchNet(const std::string& filename, const std::string& device, bool fp16, int64_t warm_batch);
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    torch::Device device;      // où tournent les calculs (GPU en pratique)
    bool fp16;                 // poids et entrées en demi-précision (sur GPU seulement)
    torch::jit::script::Module module;   // le réseau chargé du .ts
    std::mutex mutex;          // une passe à la fois (les appels sont déjà regroupés par le Batcher)
    torch::Tensor input_cpu;   // tampon d'entrée en mémoire épinglée (le GPU la lit directement), agrandi si besoin
};

// ── 10. Le regroupeur des requêtes (copie de nnue/batched_evaluator.cpp) ───
// Le GPU n'est efficace que sur de gros lots (256 positions coûtent à peine
// plus qu'une), mais un thread de recherche ne rassemble que 16 feuilles à la
// fois. Le Batcher fusionne les demandes des 16 threads :
//
//      thread 1 ── evaluate(16 positions) ─┐            (s'endort)
//      thread 2 ── evaluate(16 positions) ─┤  file   ┌──────────────┐
//         …                                ├───────► │ worker_loop  │ ─► TorchNet
//      thread 16 ─ evaluate(16 positions) ─┘  (FIFO) │ (son thread) │    (1 passe,
//                                                    └──────────────┘    ≤ 256 pos.)
//            ◄── chacun reçoit sa tranche des résultats et est réveillé
//
// Le lot part dès que (1) il y a max_batch positions en file, ou (2) tous les
// threads de recherche attendent (personne d'autre ne peut venir), ou (3)
// max_wait s'est écoulé depuis la première demande en file (un thread en
// retard ne bloque pas les autres). Section 10 du .cpp.
class Batcher final : public Evaluator {
public:
    // net : le réseau (possédé : détruit avec le Batcher) ; nb_clients : le
    // nombre de threads de recherche (condition 2 ci-dessus) ; max_batch : la
    // taille visée d'un lot fusionné (condition 1) ; max_wait : l'attente
    // maximale des retardataires avant de lancer un lot partiel (condition 3).
    // Démarre le thread du regroupeur.
    Batcher(std::unique_ptr<Evaluator> net, size_t nb_clients, size_t max_batch,
            std::chrono::microseconds max_wait = std::chrono::microseconds(500));
    ~Batcher() override;       // arrête le thread du regroupeur (après avoir servi la file)
    // Dépose la demande dans la file, réveille le regroupeur et dort jusqu'à
    // ce qu'elle soit servie.
    void evaluate(std::span<const Request> in, std::span<Result> out) override;
private:
    // La demande d'un thread. Elle vit sur la pile du thread demandeur, qui
    // dort pendant ce temps : la file ne contient que des pointeurs, rien
    // n'est copié ni alloué. `done` passe à vrai quand ses résultats sont
    // écrits dans `out` : c'est le signal du réveil.
    struct Job {
        std::span<const Request> in;
        std::span<Result> out;
        bool done = false;
    };
    void worker_loop();        // la boucle du thread du regroupeur
    std::unique_ptr<Evaluator> net;
    const size_t nb_clients, max_batch;
    const std::chrono::microseconds max_wait;
    std::mutex mutex;                    // protège tout ce qui suit jusqu'à `stop`
    std::condition_variable cv_worker;   // réveille le thread du regroupeur (nouvelle demande, arrêt)
    std::condition_variable cv_clients;  // réveille les threads de recherche servis
    std::deque<Job*> queue;              // les demandes en attente, dans l'ordre d'arrivée
    size_t queued_positions = 0;         // nombre total de positions en file (condition 1)
    size_t waiting_clients = 0;          // threads en attente, demandes vides comprises (condition 2)
    bool stop = false;                   // demandé par le destructeur
    std::vector<Request> batch_in;       // le lot fusionné (n'est touché que par le thread du regroupeur)
    std::vector<Result> batch_out;       // ses résultats, avant redistribution
    std::thread worker;                  // déclaré en dernier : démarré quand tout le reste est construit
};

// ── 8. Le cache des sorties du réseau (copie de player/nn_cache.h) ─────────
// La sortie du réseau ne dépend que du plateau et du camp au trait, donc du
// hachage Zobrist. Une même position revient souvent : par transposition
// (quand son nœud a été libéré par le ramasse-miettes), et d'un coup à
// l'autre de la partie. Le cache garde, par position, la valeur et les a
// priori (après softmax) de ses coups légaux : une position retrouvée est
// développée sans passer par le GPU. Table à accès direct : une seule place
// par indice (hachage & masque), une nouvelle entrée écrase l'ancienne.
// 64 Mo ≈ 400 000 positions. Section 8 du .cpp.
class NNCache {
public:
    // size_bytes : la mémoire visée, arrondie au nombre d'entrées puissance
    // de 2 inférieur (l'indice est alors un simple masque de bits).
    explicit NNCache(size_t size_bytes);
    // true (et value, priors[0..nb_moves)) si la position de hachage `key`
    // est en cache avec le même nombre de coups ; false sinon.
    bool lookup(uint64_t key, size_t nb_moves, float& value, float* priors);
    // Range la sortie du réseau pour cette position (écrase l'entrée en place).
    void store(uint64_t key, size_t nb_moves, float value, const float* priors);
private:
    struct Entry {
        uint64_t key = 0;                       // le hachage Zobrist complet (vérifié à la lecture)
        float    value = 0.0f;                  // la valeur du réseau, pour le camp au trait
        uint16_t nb_moves = 0;                  // vérifié aussi : protège contre une collision de hachage
        uint16_t priors[Yolah::MAX_NB_MOVES] = {};   // a priori en virgule fixe : p ≈ priors[i] / 65535
    };                                          // 164 octets (168 avec l'alignement)
    std::vector<Entry> table;                   // vide si size_bytes est trop petit : cache désactivé
    size_t mask = 0;                            // nombre d'entrées − 1
    // Verrous « en bandes » : l'entrée d'indice i est protégée par le verrou
    // i & 255. Un verrou par entrée coûterait trop de mémoire, un seul pour
    // toute la table ferait attendre les 16 threads ; 256 suffisent pour
    // qu'ils ne s'attendent presque jamais.
    std::array<std::mutex, 256> locks;
};

// ── Les réglages (ceux de l'app ; voir la liste plus haut) ─────────────────
// Les noms entre guillemets sont ceux des fichiers de configuration (.cfg).
struct Params {
    // Budgets : la recherche s'arrête au premier atteint. L'app utilise le
    // temps seul ; la vérification, 800 simulations seules (reproductible).
    uint64_t microseconds = 1'000'000;   // « microseconds » : temps par coup (0 = pas de limite de temps)
    uint64_t nb_simulations = 0;         // « nb simulations » : (0 = pas de limite)
    size_t nb_threads = 16;              // « nb threads » : threads de recherche
    size_t batch_size = 16;              // « batch size » : feuilles par thread et par appel au réseau
    size_t nn_cache_mb = 64;             // « nn cache » : taille du NNCache, en Mo
    // PUCT (section 3) : c(s) = exploration + log · ln((W_tot + base) / base).
    float cpuct_exploration = 1.0f, cpuct_exploration_log = 0.45f, cpuct_exploration_base = 500.0f;
    // Facteur σ̂ (section 3) : σ̂ = 1 + scale · (écart-type / prior − 1), l'écart-type
    // régularisé vers `prior` avec `prior_weight` visites fictives.
    float cpuct_utility_stdev_scale = 0.85f, cpuct_utility_stdev_prior = 0.40f, cpuct_utility_stdev_prior_weight = 2.0f;
    // FPU (section 3) : la pénalité d'un coup jamais essayé, ailleurs et à la racine.
    float fpu_reduction = 0.2f, root_fpu_reduction = 0.1f;
    // Softmax des logits (section 4) : T = 1, les a priori tels que le réseau les donne.
    float policy_temperature = 1.0f;
    // Choix final par la borne inférieure de confiance (section 7) : rayon
    // `lcb_stdevs` écarts-types ; candidats = au moins `min_visit_prop_for_lcb`
    // de la valeur de jeu du coup le plus visité.
    bool  use_lcb = true;
    float lcb_stdevs = 5.0f, min_visit_prop_for_lcb = 0.20f;
    // Élagage de la cible (section 7) : retire aux coups les visites qu'ils
    // n'ont eues que par l'exploration.
    bool  policy_target_pruning = true;
    // Utilité de score (« Échelle des valeurs ») : une fin de partie vaut
    // (1 − u)·signe(d) + u·tanh(d / scale).
    float score_utility = 0.05f, score_scale = 10.0f;
    // Réutiliser le graphe du coup précédent (section 2).
    bool  reuse_tree = true;
};

// ── Le bilan d'une recherche ────────────────────────────────────────────────
// Ce que la recherche rapporte de CHAQUE coup légal de la racine, une fois le
// temps écoulé. Il sert à trois choses :
//   • choisir le coup joué (play_value) ;
//   • en self-play, donner la cible de politique (target, dans le joueur
//     complet seulement) ;
//   • la vérification (test/final_mcts_check.cpp), qui compare tous ces
//     champs, coup par coup, avec ceux du joueur de l'app.
//
// D'OÙ VIENNENT LES CHAMPS. Voici la racine R d'une recherche finie et quatre
// de ses coups légaux (nombres choisis pour l'exemple ; tout ce qui en est
// déduit plus bas est calculé avec les vraies formules). Les autres coups
// légaux (a priori 0.30 à eux tous) n'ont jamais été essayés, comme h1-h3. Noir est au trait
// en R ; W et Q sont donc du point de vue de Noir (le camp qui JOUE le coup
// menant à l'enfant, glossaire).
//
//                                     R          Noir au trait
//                                     │          1 520 visites des coups
//         ┌──────────────────┬────────┴─────────┬──────────────────┐
//       c3-c6              e4-e7              b2-b5              h1-h3   ┐
//     P = 0.40           P = 0.20           P = 0.09           P = 0.01  │ ARÊTES
//     N(R,·) = 1 000     N(R,·) = 400       N(R,·) = 120       N(R,·) = 0┘ (Edge)
//         ▼                  ▼                  ▼                  ▼
//     ┌─────────┐        ┌─────────┐        ┌─────────┐       (aucun nœud :
//     │N = 1 000│        │N =   400│        │N =   120│        jamais pris)
//     │W = +100 │        │W = +80  │        │W = +6   │
//     │W2 = 170 │        │W2 = 80  │        │W2 = 19.5│       NŒUDS (Node)
//     └─────────┘        └─────────┘        └─────────┘
//
//   move, prior, visits : lus sur l'ARÊTE (Edge::action, Edge::prior, Edge::n).
//   q = W / N, q2 = W2 / N : lus sur le NŒUD enfant (Node::w, w2, n) ; ce sont
//       les moyennes des valeurs remontées et de leurs carrés :
//         c3-c6 : q = 100 / 1 000 = 0.10   q2 = 170 / 1 000 = 0.17
//         e4-e7 : q =  80 /   400 = 0.20   q2 =  80 /   400 = 0.20
//         b2-b5 : q =   6 /   120 = 0.05   q2 = 19.5 /  120 = 0.1625
//       d'où l'écart-type des valeurs √(q2 − q²) = √(0.17 − 0.01) = 0.40
//       pour c3-c6, et aussi 0.40 pour les deux autres (choisi ainsi).
//       h1-h3 n'a pas de nœud : q = q2 = 0.
//   lcb, target, play_value : CALCULÉS ensuite par root_play_values (section 7
//       du .cpp), à partir des champs précédents ; détail à play_value.
//
// Le bilan obtenu (le vecteur SearchResult::children, trié par play_value) :
//     coup   prior  visits    q      q2      lcb     target  play_value
//     e4-e7  0.20      400  +0.20  0.200   +0.100      400     2 778    ← joué
//     c3-c6  0.40    1 000  +0.10  0.170   +0.037    1 000     1 000
//     b2-b5  0.09      120  +0.05  0.163   −0.133       75        75    (élagué)
//     h1-h3  0.01        0   0     0       −2           0         0     (jamais essayé)
//     …      (les autres coups jamais essayés : comme h1-h3)
// Le plus visité (c3-c6) n'est pas joué : e4-e7 a une moyenne plus haute,
// et assez de visites pour qu'elle soit sûre (la LCB, plus bas).
struct ChildStat {
    Move  move;
    // P(racine, coup) : l'a priori de la tête de politique, après le softmax
    // sur les seuls coups légaux (somme 1 sur la racine). Ce que le réseau
    // pensait AVANT la recherche.
    float prior;
    // N(racine, coup) : visites TERMINÉES passées par cette ARÊTE (glossaire),
    // y compris celles héritées du coup précédent quand le graphe a été
    // réutilisé. Le nœud enfant peut en avoir plus (son N) : dans l'ancien
    // graphe, il a pu être atteint aussi par d'autres parents, libérés depuis
    // par le ramasse-miettes, mais ses visites, elles, sont restées.
    uint32_t visits;
    // Q du NŒUD enfant = W / N de l'enfant : la valeur moyenne de tout ce qui
    // y a été remonté, par n'importe quel chemin. Du point de vue du camp au
    // trait à la RACINE (celui qui joue ce coup) : plus grand = meilleur pour
    // nous. Dans [−1, 1] ; 0 si l'enfant n'a aucune visite (ou n'existe pas).
    float q;
    // Moyenne des CARRÉS des valeurs remontées dans l'enfant (W2 / N). Avec q,
    // elle donne la variance des valeurs, par la formule de König-Huygens
    // (« moyenne des carrés moins carré de la moyenne »). Avec v_1 … v_N les
    // valeurs remontées, q = (1/N) Σ v_i et q2 = (1/N) Σ v_i² :
    //     variance = (1/N) Σ (v_i − q)²                       (définition)
    //              = (1/N) Σ v_i²  −  2q · (1/N) Σ v_i  +  q²  (développer le carré)
    //              = q2 − 2q · q + q²
    //              = q2 − q²
    //     (c3-c6 : 0.17 − 0.10² = 0.16, écart-type 0.40)
    // On n'a donc pas besoin de garder les valeurs une à une : deux sommes,
    // W et W2, suffisent, et une remontée n'ajoute qu'à ces deux sommes.
    // La LCB (section 7) en tire l'incertitude sur la MOYENNE q, qui diminue
    // avec le nombre n de visites (n = visits) :
    //     σ(Q) = √(variance / n)   (c3-c6 : √(0.16 / 1 000) = 0.0126 ; × 5 = 0.063)
    // Le code y ajoute un a priori de poids 1/n² (une « visite fictive » de
    // variance 1 de plus), pour qu'un coup à 1 ou 2 visites n'ait pas une
    // variance nulle donc une borne absurdement sûre : à n = 1, la variance
    // vaut alors 0.5 au lieu de 0, la visite fictive compte comme une visite
    // de plus (n effectif 2), σ(Q) = √(0.5 / 2) = 0.5 et le rayon 5·σ(Q) =
    // 2.5, toute l'échelle ; à n = 1 000, l'a priori pèse 10⁻⁶ et ne change
    // rien.
    float q2;
    // Borne inférieure de confiance (« lower confidence bound ») :
    //     lcb = q − 5 · σ(Q)
    // q est une MOYENNE de n = visits valeurs ; si l'on refaisait la
    // recherche, on obtiendrait une autre moyenne. σ(Q) mesure de combien
    // elle varierait : c'est l'écart-type de la moyenne, pas celui des
    // valeurs. Si les n valeurs étaient indépendantes, de variance σ² :
    //     Var(q) = Var((1/n) Σ v_i) = (1/n²) · Σ Var(v_i) = (1/n²) · n·σ² = σ² / n
    // donc σ(Q) = σ / √n : quatre fois plus de visites, une incertitude deux
    // fois plus petite. σ² est la variance q2 − q² ci-dessus (avec son a
    // priori, qui ne compte que pour n très petit).
    // Le 5 (lcb_stdevs, la valeur de KataGo) fixe la prudence : la borne est
    // 5 écarts-types sous la moyenne, donc « le coup vaut au moins lcb » avec
    // une très grande confiance. Ce n'est pas une vraie probabilité : les
    // valeurs d'une recherche ne sont ni indépendantes (les visites suivent
    // les mêmes chemins) ni gaussiennes ; 5 est une marge de sécurité large,
    // pour qu'un coup ne gagne par la LCB que si son avance est nette.
    // Quels coups ont une lcb. La LCB se fait en deux temps (section 7) :
    //   1. CHOISIR le gagnant, parmi les seuls « candidats » (au moins 20 % de
    //      la valeur de jeu du plus visité) : celui de plus grande lcb ;
    //   2. lui donner une valeur de jeu assez grande pour passer devant TOUS
    //      les autres coups, candidats ou non ; pour chacun, le calcul se sert
    //      de SA lcb (l'écart entre sa borne et celle du gagnant).
    // Il faut donc la lcb de tous les coups visités, pas seulement celle des
    // candidats. Dans l'exemple, b2-b5 (75 < 200) n'est pas candidat, mais sa
    // lcb −0.133 sert à l'étape 2 : elle donne 3.28 × 75 = 246.
    // −2 (impossible, hors de [−1, 1]) veut dire « pas de lcb » :
    //   • coup jamais visité : pas de moyenne, donc pas de borne (h1-h3) ;
    //   • LCB désactivée (use_lcb = false) ou aucune visite à la racine :
    //     l'étape LCB n'a pas lieu, tous les coups gardent −2.
    float lcb;
    // LA « VALEUR DE JEU » : le nombre qui décide du coup joué (le plus grand).
    //
    // ── L'idée ──────────────────────────────────────────────────────────────
    // Point de départ, la règle d'AlphaZero : jouer le coup le PLUS VISITÉ.
    // PUCT envoie la plupart des visites vers les coups qui se révèlent bons ;
    // compter les visites est plus sûr que prendre le meilleur Q, car un Q
    // élevé sur peu de visites peut n'être que de la chance.
    // Mais les visites ont deux défauts, que KataGo corrige l'un après l'autre :
    //   1. Une partie des visites ne vient pas de la valeur du coup mais du
    //      bonus d'exploration : un coup a pu être beaucoup visité parce qu'il
    //      PARAISSAIT bon au début, avant que son Q ne baisse. ÉLAGAGE : on
    //      retire à chaque coup les visites que PUCT ne lui donnerait plus,
    //      vu ce qu'on sait à la fin.
    //   2. Les visites réagissent lentement : un coup reconnu meilleur tard a
    //      encore moins de visites que l'ancien favori. LCB : si un coup
    //      assez visité a une moyenne nettement plus haute ET sûre (sa borne
    //      basse dépasse celle des autres), on le fait passer devant.
    // Le résultat sert seulement à CLASSER les coups : ce n'est plus un nombre
    // de visites. Sur l'exemple : c3-c6 est le plus visité (1 000), mais
    // e4-e7 (400 visites, Q 0.20 contre 0.10) est joué grâce à la LCB. Et
    // b2-b5 a reçu 120 visites, mais avec sa moyenne finale (Q = 0.05, bien
    // en dessous des deux autres), PUCT ne lui en aurait donné que 75 : les
    // 45 autres, il ne les a eues que parce qu'il paraissait meilleur au
    // début. L'élagage ne lui compte donc que 75 (calcul au point c).
    // Ici cela ne change pas le coup joué ; cela change la cible de politique
    // en self-play, et cela peut changer le choix : la LCB (point d) ne
    // choisit que parmi les « candidats », les coups dont la valeur de jeu
    // APRÈS élagage atteint au moins 20 % de celle du plus visité (ici
    // 20 % × 1 000 = 200). Un coup de 250 visites élagué à 150 n'est plus
    // candidat : la LCB ne peut plus le faire jouer, même si sa borne est la
    // meilleure. (b2-b5 n'est candidat ni avant, 120, ni après, 75.)
    //
    // ── Les détails, sur l'exemple ──────────────────────────────────────────
    // La racine de l'exemple (en tête de ChildStat), avec ce dont les calculs
    // ont besoin : sur chaque arête l'a priori P et les visites N(R, ·) ; dans
    // chaque nœud enfant sa moyenne Q et l'écart-type σ de ses valeurs.
    //
    //                                     R          Noir au trait
    //                                     │          N(R) = 1 521
    //         ┌──────────────────┬────────┴─────────┬──────────────────┐
    //       c3-c6              e4-e7              b2-b5              h1-h3
    //     P = 0.40           P = 0.20           P = 0.09           P = 0.01
    //     N(R,·) = 1 000     N(R,·) = 400       N(R,·) = 120       N(R,·) = 0
    //         ▼                  ▼                  ▼
    //     ┌─────────┐        ┌─────────┐        ┌─────────┐       (jamais pris)
    //     │Q = 0.10 │        │Q = 0.20 │        │Q = 0.05 │
    //     │σ = 0.40 │        │σ = 0.40 │        │σ = 0.40 │
    //     └─────────┘        └─────────┘        └─────────┘
    //
    // a) Les grandeurs nécessaires (section 3 du .cpp).
    //    Hypothèses : 1 520 visites des coups de la racine (1 000 + 400 + 120
    //    + 0) ; dans chaque enfant, l'écart-type des valeurs remontées vaut
    //    0.40 (√(q2 − q²), voir q2 ; 0 si toutes les visites ont ramené la même
    //    valeur, 1 au plus : moitié +1, moitié −1).
    //    Le coefficient du bonus d'exploration de PUCT, E = c · σ̂ · √W_tot :
    //      • √W_tot = √(1 520 + 0.01) = 38.99 : l'exploration pèse plus quand
    //        la racine a plus de visites (moins vite qu'elles : la racine carrée) ;
    //      • c = 1 + 0.45 · ln((W_tot + 500) / 500) = 1 + 0.45 · ln(4.04) = 1.628 :
    //        la constante d'exploration de KataGo, qui grandit doucement ;
    //      • σ̂ (« sigma chapeau » ; le chapeau note une grandeur ESTIMÉE à
    //        partir des données) : le facteur d'écart-type de KataGo,
    //            σ̂ = 1 + 0.85 · (écart-type à la racine / 0.40 − 1),
    //        qui adapte l'exploration à l'incertitude de la position :
    //        écart-type 0.8 (valeurs qui se contredisent) → σ̂ = 1.85, on explore
    //        presque deux fois plus ; 0.05 (tout concorde, position « jouée
    //        d'avance ») → σ̂ = 0.26, on explore peu ; 0.40 (la référence de
    //        KataGo) → σ̂ = 1.
    //        L'écart-type à la racine porte sur les n = N(R) = 1 521 valeurs
    //        remontées dans R (celles des coups, plus la première visite, qui a
    //        évalué R elle-même), de moyenne m = W(R)/n et de moyenne des carrés
    //        moy(v²) = W2(R)/n, « régularisé » vers 0.40 par 2 visites fictives
    //        (utility_stdev_factor) :
    //            écart-type² = ((m² + 0.40²) · 2 + moy(v²) · n) / (2 + n − 1) − m²
    //        Après une ou deux visites, il reste proche de 0.40 (pas de σ̂
    //        absurde) ; à n = 1 521, les visites fictives ne pèsent presque rien
    //        et l'on retrouve la variance moy(v²) − m². Ici les valeurs de R
    //        mélangent celles des enfants : variance = 0.40² (dans les enfants)
    //        + 0.002 (entre leurs moyennes 0.10, 0.20, 0.05) ≈ 0.162, écart-type
    //        ≈ 0.403, σ̂ = 1.006 ≈ 1.
    //    D'où E = 1.628 · 1 · 38.99 = 63.5.
    //
    // b) Départ : play_value = visits, soit 1 000 / 400 / 120 / 0.
    //    Le coup de référence b est « le plus visité », mesuré par
    //    v · (v − 1) / v + 2 · P (une visite de moins, la dernière pouvant être
    //    aberrante, et l'a priori pour départager) : c3-c6.
    //
    // c) Élagage de la cible. Le score PUCT final de b :
    //        score(b) = Q_b + E · P_b / (1 + N_b) = 0.10 + 63.5 · 0.40 / 1 001 = 0.125
    //    Pour un autre coup de moyenne Q et d'a priori P, PUCT ne le préférerait
    //    à b que tant que son score Q + E · P / (1 + n) dépasse 0.125, soit pour
    //    n < E · P / (score(b) − Q) − 1. Il garde le plus petit de ce n (arrondi
    //    au-dessus) et de ses visites :
    //        b2-b5 : 63.5 · 0.09 / (0.125 − 0.05) − 1 = 74.8 → 75 (au lieu de 120)
    //        e4-e7 : Q = 0.20 ≥ 0.125, son Q seul le met devant b : rien n'est
    //                retiré (400)
    //        h1-h3 : jamais visité : 0
    //    → 1 000 / 400 / 75 / 0, qui est aussi `target`.
    //
    // d) LCB. Pour chaque coup visité, sa borne basse lcb = Q − 5 · σ(Q), avec
    //    σ(Q) = écart-type / √visites (voir lcb) ; le rayon r = 5 · σ(Q) :
    //        c3-c6 : r = 5 · 0.40 / √1 000 = 0.063   lcb = 0.10 − 0.063 = 0.037
    //        e4-e7 : r = 5 · 0.40 / √400   = 0.100   lcb = 0.20 − 0.100 = 0.100
    //        b2-b5 : r = 5 · 0.40 / √120   = 0.183   lcb = 0.05 − 0.183 = −0.133
    //    Candidats : au moins 20 % de la valeur de jeu de b, ici ≥ 200 : c3-c6
    //    et e4-e7 (b2-b5, 75, a trop peu de visites pour qu'on se fie même à
    //    sa borne). Gagnant : le candidat de plus grande lcb, e4-e7.
    //    Le gagnant reçoit une valeur de jeu assez grande pour passer devant
    //    CHAQUE autre coup i visité : pour i, d'écart des bornes
    //    e = lcb(gagnant) − lcb(i) et de rayon r_i,
    //        facteur_i = ((r_i + e) / (r_i + 0.2 · e))²,   besoin_i = facteur_i × play_value_i
    //        c3-c6 : e = 0.063 : ((0.063 + 0.063) / (0.063 + 0.013))² = 2.78 → 2 778
    //        b2-b5 : e = 0.233 : ((0.183 + 0.233) / (0.183 + 0.047))² = 3.28 →   246
    //    play_value(e4-e7) = max(400, 2 778, 246) = 2 778 > 1 000 : e4-e7 joué.
    //    Le facteur vaut 1 quand les bornes sont égales et monte jusqu'à 5² =
    //    25 quand l'écart est très grand devant le rayon (formule de KataGo) :
    //    un gagnant à peine devant ne reçoit qu'un peu plus que l'autre coup.
    float play_value;
    // Les visites après la correction 1 seule (avant la LCB).
    // Dans le joueur de l'app, il ne sert qu'à UNE chose : si Σ target = 0,
    // la recherche n'a rien appris d'utile, et make_result joue le coup de
    // plus grand prior au lieu de children[0].
    // Pourquoi ne pas tester Σ visits = 0 à la place : les deux diffèrent
    // dans un cas limite, quand aucun coup n'a plus d'une visite (recherche
    // presque nulle, temps minuscule). Le « meilleur » b de l'élagage peut
    // alors être un coup jamais visité (à 0 ou 1 visite, c'est l'a priori qui
    // départage, section 7), et l'élagage ramener à 0 les coups visités une
    // fois : Σ target = 0 alors que Σ visits > 0. Le garder reproduit le
    // joueur de l'app exactement, cas limite compris.
    // Son vrai rôle est ailleurs, dans le joueur complet
    // (player/alphazero_mcts.cpp) et pas dans cette copie : en self-play,
    // target / Σ target est la cible de politique qu'apprend le réseau (« où
    // la recherche a vraiment voulu aller », sans les visites dues à
    // l'exploration), et le coup est tiré au sort ∝ target^(1/τ) pendant les
    // premiers plis.
    float target;
};
struct SearchResult {
    // Le coup choisi : children[0].move (le plus grand play_value), ou le coup
    // de plus grand prior si aucune visite n'a servi. Move::none() est aussi
    // la PASSE : c'est le coup rendu quand la passe est le seul coup légal.
    Move best_move = Move::none();
    // Simulations TERMINÉES pendant cette recherche : toutes les descentes
    // remontées, qu'elles aient fini sur une feuille évaluée par le réseau,
    // une position du cache, une fin de partie, ou en s'ajoutant à une feuille
    // déjà dans le lot. Les descentes annulées (collisions) ne comptent pas,
    // ni les visites héritées du coup précédent.
    uint64_t nb_simulations = 0;
    // Positions réellement envoyées au réseau par les threads (hors cache, et
    // hors l'évaluation de la racine faite avant eux). ≤ nb_simulations : une
    // évaluation sert au moins une descente. L'écart mesure ce qu'apportent le
    // cache et les fins de partie.
    uint64_t nb_evaluations = 0;
    // Un ChildStat par coup légal de la racine (aussi ceux jamais visités),
    // triés par play_value décroissante, puis par q ; à égalité complète,
    // dans l'ordre de Yolah::moves.
    std::vector<ChildStat> children;
};

// ── La recherche (sections 1 à 7 du .cpp) ──────────────────────────────────
// Possède le graphe (les nœuds et leurs arêtes, dans le pool mémoire
// `memory`, et la table qui retrouve un nœud par sa position) et le NNCache.
// Utilise l'Evaluator qu'on lui donne, sans le posséder. Le graphe survit d'un
// appel à search() au suivant : c'est la réutilisation (section 2) ; reset()
// l'efface. Le NNCache survit même à reset() (il ne dépend que du réseau).
class Search {
public:
    // evaluator : le réseau, ou le Batcher devant lui (plusieurs threads) ;
    // params : les réglages, copiés (lève std::invalid_argument si les deux
    // budgets, temps et simulations, sont nuls : la recherche ne finirait
    // jamais ; nb_threads et batch_size sont portés à 1 au moins) ; memory :
    // la ressource mémoire où vivent les nœuds et leurs arêtes. Crée le
    // NNCache (nn_cache_mb Mo).
    Search(Evaluator& evaluator, const Params& params, std::pmr::memory_resource* memory);
    ~Search();                                 // libère tout le graphe
    // Une recherche complète depuis `state`, qui ne doit pas être finie :
    //   1. la racine : retrouvée dans le graphe du coup précédent (avec toutes
    //      ses visites) si reuse_tree, le reste du graphe étant libéré ; sinon
    //      un graphe neuf. Si elle n'a jamais été évaluée : un appel au réseau,
    //      seule, et développement ;
    //   2. nb_threads threads run_worker jusqu'à la fin du budget (un seul
    //      thread : dans le thread appelant, sans en créer) ;
    //   3. le bilan (make_result) : un ChildStat par coup, le coup choisi.
    // Bloquant : rend la main quand tous les threads ont fini.
    SearchResult search(const Yolah& state);
    // Oublie tout le graphe (à la fin d'une partie : les positions de la
    // suivante n'auront rien à voir). Le cache reste.
    void reset();
private:
    struct Node;          // une position : N, W, W2, pending, état, arêtes (section 1)
    struct Edge;          // un coup : a priori P, N(s, a), pending, nœud d'arrivée (section 1)
    struct Worker;        // les tampons d'un thread : feuilles du lot, chemins des descentes, requêtes (section 1)
    struct PathStep;      // un pli d'une descente : (nœud atteint, arête prise pour l'atteindre) (section 1)
    // Le cycle de vie d'un nœud (Node::state, atomique) :
    //
    //     UNEXPANDED ──(un thread le prend, échange atomique)──► EXPANDING
    //         │                                                     │
    //         │ partie finie                       réponse du réseau │
    //         ▼                                    ou trouvé en cache ▼
    //      TERMINAL                                              EXPANDED
    //
    //   • UNEXPANDED : créé (une arête y mène) mais jamais évalué ;
    //   • EXPANDING : un thread l'a pris et attend la réponse du réseau. Une
    //     autre descente qui y arrive s'y joint si c'est une feuille de son
    //     propre lot, s'annule sinon (collision, section 6) ;
    //   • EXPANDED : ses arêtes existent (une par coup légal), on descend au
    //     travers. Ne change plus ;
    //   • TERMINAL : la partie y est finie ; sa valeur exacte est remontée à
    //     chaque descente qui y arrive, sans réseau. Ne change plus.
    // L'échange UNEXPANDED → EXPANDING est atomique (compare_exchange) : si
    // deux threads arrivent ensemble, un seul évalue le nœud.
    enum : uint8_t { UNEXPANDED = 0, EXPANDING = 1, EXPANDED = 2, TERMINAL = 3 };
    // La table des nœuds (clé de position → nœud), qui fait du graphe un
    // graphe : une position déjà créée par un autre chemin y est retrouvée.
    // Elle est coupée en 64 tranches, chacune avec son verrou ; la tranche
    // d'une clé est clé % 64. Deux threads ne s'attendent que s'ils touchent
    // la même tranche au même moment (1 chance sur 64), alors qu'un seul
    // verrou pour toute la table ferait attendre les 16 threads. La table
    // n'est touchée que lorsqu'une arête est prise pour la première fois
    // (ensuite, Edge::child donne le nœud directement).
    static constexpr size_t NB_SHARDS = 64;
    struct Shard {                             // une tranche de la table des nœuds
        std::mutex mutex;
        std::unordered_map<uint64_t, Node*> nodes;   // clé = node_key (Zobrist + score de Noir, section 2)
    };
    // ── section 2 : le graphe ──
    Node* new_node();                          // un nœud vide (UNEXPANDED), alloué dans le pool
    Node* lookup(uint64_t key);                // le nœud de cette clé, ou nullptr (sous le verrou de sa tranche)
    void  destroy_all();                       // libère tous les nœuds, vide la table, plus de racine
    // Ramasse-miettes « marquer et balayer », après un changement de racine :
    void  mark(Node* node);                    // marque node et tout ce qu'il atteint (parcours en profondeur)
    void  collect(Node* keep);                 // marque depuis keep, puis libère tout nœud non marqué
    // La position `key` est-elle déjà dans le graphe ? Si oui, elle devient
    // la racine, avec ses statistiques, et le reste est libéré (collect).
    bool  reuse(uint64_t key);
    void  new_root(uint64_t key);              // efface tout ; graphe neuf réduit à une racine UNEXPANDED
    void  expand_root(const Yolah& state);     // évalue (seule) et développe la racine, avant les threads
    // ── section 6 : la boucle d'un thread ──
    // Rassembler un lot de descentes, l'évaluer, développer, remonter ; jusqu'à
    // ce que `stop` passe à vrai. w : les tampons de ce thread ; root_state :
    // la position de la racine (chaque descente en part et la rejoue).
    void  run_worker(Worker& w, const Yolah& root_state);
    // ── section 3 : le choix d'un coup pendant la descente ──
    // L'indice de l'arête de node au score PUCT maximal, Q + E·P/(1 + N(s, a)),
    // descentes en cours comprises ; is_root : la FPU de la racine est plus
    // généreuse (0.1 au lieu de 0.2).
    uint32_t select_child(const Node& node, bool is_root) const;
    // σ̂ du nœud (section 3) : > 1 si les valeurs remontées se contredisent
    // (on explore plus), < 1 si elles concordent ; visits = N(node).
    float utility_stdev_factor(const Node& node, uint32_t visits) const;
    // La valeur supposée d'un coup jamais essayé depuis node : ≈ la valeur de
    // node, moins une pénalité qui croît avec visited_policy_mass (la somme
    // des a priori des coups déjà essayés) ; visits = N(node).
    float fpu_value(const Node& node, uint32_t visits, float visited_policy_mass, bool is_root) const;
    // ── section 4 : le développement ──
    // Le softmax des logits des coups légaux (température policy_temperature) :
    // priors[0..nb_moves), somme 1.
    void  priors_from_logits(const Result& result, size_t nb_moves, float* priors) const;
    // Crée les arêtes de node (une par coup, avec son a priori, sans nœud
    // d'arrivée), retient la valeur du réseau (net_v, pour la FPU) et publie
    // le tout en passant node à EXPANDED.
    void  expand(Node& node, const Yolah::MoveList& moves, const float* priors, float value);
    // ── section 5 : la remontée ──
    // La valeur d'une feuille (pour le camp au trait à la feuille) ajoutée à
    // chaque nœud du chemin, avec un changement de signe par pli ; N + 1 pour
    // chaque nœud et chaque arête ; retire les « pending » posés à la descente.
    void  backup(const PathStep* path, size_t length, float leaf_value);
    // Une descente annulée (collision) : retire ses « pending », ne remonte rien.
    void  undo_pending(const PathStep* path, size_t length);
    // La valeur exacte d'une fin de partie pour le camp au trait :
    // 0.95·signe(d) + 0.05·tanh(d / 10), d = son écart de points.
    float terminal_value(const Yolah& state) const;
    // ── section 7 : le bilan de la racine ──
    void  root_play_values(std::vector<ChildStat>& children) const;           // play_value, target, lcb (élagage + LCB)
    SearchResult make_result();                // les ChildStat de la racine, triés, et le coup choisi

    Evaluator& evaluator;
    Params prm;                                // les réglages (copie, corrigée par le constructeur)
    // L'allocateur des nœuds : il prend sa mémoire dans le « pool » du Player
    // (de grands blocs découpés en nœuds) : allocations et libérations
    // rapides, sans passer par malloc à chaque nœud.
    std::pmr::polymorphic_allocator<> alloc;
    Shard shards[NB_SHARDS];                   // la table des nœuds
    Node* root = nullptr;                      // la racine de la recherche en cours (ou de la dernière ; nullptr = pas de graphe)
    std::unique_ptr<NNCache> cache;
    // Le numéro du marquage en cours (ramasse-miettes) : un nœud est vivant
    // si son Node::mark vaut ce numéro ; incrémenté à chaque collect, ce qui
    // évite d'effacer les anciennes marques.
    uint32_t mark_counter = 0;
    // Compteurs partagés par les threads pendant une recherche (remis à 0 au
    // début de chaque search()) :
    std::atomic<uint64_t> simulations{0};      // descentes terminées (→ SearchResult::nb_simulations)
    // Budget en simulations seulement (la vérification : 800) : chaque thread
    // RÉSERVE ici ses descentes avant de les faire, pour qu'à eux tous ils ne
    // dépassent jamais le budget, et rend celles qu'il n'a pas faites.
    std::atomic<uint64_t> reserved{0};
    std::atomic<uint64_t> evaluations{0};      // positions envoyées au réseau (→ SearchResult::nb_evaluations)
    // Le signal d'arrêt : mis à vrai par le premier thread qui voit le budget
    // épuisé (l'horloge est regardée une fois par lot) ; les autres finissent
    // leur lot en cours et s'arrêtent.
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point deadline;   // début de search() + microseconds
};

// ── Le joueur (copie de player/alphazero_mcts_player.cpp, réglages de l'app) ─
// Assemble les morceaux : le réseau (TorchNet), le regroupeur devant lui s'il
// y a plusieurs threads, et la recherche. C'est ce que l'app appelle à chaque
// coup du joueur « MCTS ResNet ».
class Player {
public:
    // weights : le .ts ; nb_threads > 1 : les requêtes passent par un Batcher
    // (lots fusionnés de nb_threads × batch_size = 256 positions au plus) ;
    // fp16 et device : comme TorchNet.
    Player(const std::string& weights, const Params& params, bool fp16 = true, const std::string& device = "cuda");
    ~Player();
    Move play(const Yolah& yolah);              // une recherche, et son coup
    void game_over();                           // à appeler à la fin de chaque partie : oublie le graphe (le cache reste)
    const SearchResult& last_result() const { return last; }   // le bilan de la dernière recherche (vérification, analyse)
private:
    // L'ordre des membres compte : ils sont détruits dans l'ordre INVERSE. Le
    // pool `memory` est donc détruit après `searcher`, dont les nœuds vivent
    // dedans, et `net` après `searcher`, qui s'en sert.
    std::pmr::synchronized_pool_resource memory;   // la mémoire des nœuds (« synchronized » : sûre entre threads)
    std::unique_ptr<Evaluator> net;                // TorchNet, ou Batcher devant un TorchNet
    std::unique_ptr<Search> searcher;
    SearchResult last;
};

} // namespace final_mcts

#endif
