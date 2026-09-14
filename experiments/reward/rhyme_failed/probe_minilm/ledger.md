# Ledger — probe_minilm / 20260902_2343_e3_probe

```
==========================================================================================================================================================================
THE LEDGER  —  predictions (Tier 0/1) vs the trained model (Tier 2)
experiment: probe_minilm / 20260902_2343_e3_probe
==========================================================================================================================================================================
question                   | predicted                                            | observed                                                 | verdict
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
how easy is this task      | best baseline 0.998 (lexical_nb); separability 0.989 | val acc 0.9873 (se 0.0014), best at step 3500 of 4095    | model BELOW surface ceiling — investigate
length shortcut            | char_length baseline 0.946; P(longer wins) 0.943; near-equal pairs 0.107 | worst length-matched slice: match_word_count acc 0.9544 (n=526, se 0.0091) | DROPS on length-controlled slice — length shortcut in use
surface-reader or not      | frac_surface_controlled 0.004; separability_min 0.981 (probe worst slice) | model worst slice: match_word_count acc 0.9544 (n=526)   | read the slice table
did it learn the PC1 axis  | Tier-1 PC1 explains 0.054 of delta var (the rejection-strategy axis) | cos(w, PC1) +0.735, cos(w, mean delta) +0.825, own-space PC1 frac 0.06 | DIRECT comparison — spaces verified identical (model==instrument): cos(w, Tier-1 artifact axis) as stated
margin inflation on separable data | predicted: margins/|r| grow after accuracy saturates (step-13 demo) | abs_reward_max 5.89 -> 10.32 across the run; best checkpoint (step 3500) predates most inflation | CONFIRMED
honest evaluation          | re-split: prompt overlap 0.0%, shift real            | eval on 6289 out-of-sample-prompt pairs                  | every number above is leakage-free
run facts                  | budget: 4095 steps                                   | early_stopped=False at step 4095, wall 179s              | ran full budget
==========================================================================================================================================================================
```
