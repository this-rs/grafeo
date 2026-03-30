# TP.3 — Benchmark E2E Manuel

## Setup

```bash
# Persona vierge pour chaque session
rm -rf /tmp/tp3_persona && mkdir -p /tmp/tp3_persona

# Lancer obrain-chat
/tmp/obrain-chat/dist/obrain-chat-macos-arm64 \
  --model ~/models/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf \
  --n-ctx 8192 --kv-capacity 8192 \
  --persona /tmp/tp3_persona 2>&1 | tee /tmp/tp3_session.log
```

## Session 1 — Convergence & Reward (7 turns)

Poser ces questions dans l'ordre, noter les logs `[Reward]`, `[AFE]`, `[T4]` :

```
1. Bonjour, je m'appelle Thomas
2. Je suis developpeur Rust et je travaille sur des bases de donnees graphe
3. Mon projet principal s'appelle Obrain, c'est une base cognitive
4. Merci, c'est tres clair
5. Peux-tu me rappeler ce que je t'ai dit sur moi ?
6. Quel est ton reward moyen et combien de memoires as-tu retenues ?
7. /quit
```

**Criteres** :
- [ ] M1: Reward augmente entre turn 2 et turn 5 (engagement)
- [ ] M2: Turn 4 ("merci") produit reward > 0
- [ ] M3: Turn 5 rappelle le prenom et/ou le projet
- [ ] M4: `[AFE]` apparait au moins 1 fois (mutation triggered)
- [ ] M5: `[T4]` affiché au shutdown (token polarities saved)

## Session 2 — Reformulation Detection (5 turns)

```bash
rm -rf /tmp/tp3_persona && mkdir -p /tmp/tp3_persona
# relancer obrain-chat...
```

```
1. Explique-moi ce qu'est un KV cache dans un LLM
2. Redis-moi ce qu'est un KV cache dans un LLM
3. Explique le KV cache des LLM
4. Merci, maintenant parle-moi de la quantization
5. /quit
```

**Criteres** :
- [ ] M6: Turns 2 et 3 ont un reward négatif (reformulation penalty)
- [ ] M7: Turn 4 a un reward positif (changement de sujet + "merci")

## Session 3 — Self-Awareness (4 turns)

```bash
rm -rf /tmp/tp3_persona && mkdir -p /tmp/tp3_persona
# relancer obrain-chat, faire 3 turns de context d'abord
```

```
1. Bonjour
2. Comment ca va ?
3. Parle-moi de toi, comment tu fonctionnes ?
4. /quit
```

**Criteres** :
- [ ] M8: Le modele mentionne "formule" ou "attention" ou "reward" dans sa reponse (Self nodes retrieves)

## Rapport Go/No-Go

Copier les lignes `[Reward]`, `[AFE]`, `[T4]`, `[PersistNet]` des 3 sessions
et les partager. Je redige le rapport.

### Go si :
- M1 + M2 + M3 : le reward structurel fonctionne ✓
- M4 : l'evolution de formules se declenche ✓
- M5 : le T4 token polarity persiste ✓
- M6 + M7 : la detection de reformulation fonctionne ✓

### No-Go si :
- Aucun reward positif sur 7 turns
- Crash ou segfault
- Le modele ne rappelle rien (PersistNet defaillant)
