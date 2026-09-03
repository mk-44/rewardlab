# Ledger — distill_bert / 20260902_0033_e2_base

```
==========================================================================================================================================================================
THE LEDGER  —  predictions (Tier 0/1) vs the trained model (Tier 2)
experiment: distill_bert / 20260902_0033_e2_base
==========================================================================================================================================================================
question                   | predicted                                            | observed                                                 | verdict
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
how easy is this task      | best baseline 0.998 (lexical_nb); separability 0.989 | val acc 0.9997 (se 0.0002), best at step 1000 of 3000    | CONFIRMED — at the surface ceiling almost immediately
length shortcut            | char_length baseline 0.946; P(longer wins) 0.943; near-equal pairs 0.107 | worst length-matched slice: match_char_length acc 0.9987 (n=745, se 0.0013) | survives length control — NOT length-dependent
surface-reader or not      | frac_surface_controlled 0.004; separability_min 0.981 (probe worst slice) | model worst slice: match_newline_count acc 0.9979 (n=937) | holds on every matched slice — beyond any single surface feature
did it learn the PC1 axis  | Tier-1 PC1 explains 0.054 of delta var (the rejection-strategy axis) | cos(w, PC1) +0.923, cos(w, mean delta) +0.924, own-space PC1 frac 0.89 | own-space numbers only — spaces DIFFER, not comparable to Tier-1 axes (step-20 F1)
margin inflation on separable data | predicted: margins/|r| grow after accuracy saturates (step-13 demo) | abs_reward_max 7.69 -> 9.15 across the run; best checkpoint (step 1000) predates most inflation | CONFIRMED
honest evaluation          | re-split: prompt overlap 0.0%, shift real            | eval on 6289 out-of-sample-prompt pairs                  | every number above is leakage-free
run facts                  | budget: 5458 steps                                   | early_stopped=True at step 3000, wall 942s               | early stop reclaimed the saturated tail
==========================================================================================================================================================================
```
