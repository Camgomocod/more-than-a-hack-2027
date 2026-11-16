# Held-out IDSSE benchmark

Matches: J03WN1, J03WPY
Bootstrap draws: 500

Panel A is the fair IDSSE-only comparison. Panel B transfers BAR-ATH candidates with Smart Tagging neutralised; treat B as indicative.

## Per match

 match panel            model  spell_f1  spell_lo  spell_hi  macro_f1  alive_dead_f1  n_gt_spells  n_pred_spells  n_graded  transitions_25
J03WN1     A baseline_nearest  0.139501  0.129678  0.149477  0.660156       0.821201          500           1986    141561        0.361121
J03WN1     A     hmm_physical  0.372881  0.337022  0.407288  0.731225       0.852449          500            326    141561        0.369942
J03WN1     A   hmm_bundesliga  0.365133  0.331719  0.400000  0.699961       0.826292          500            326    141561        0.366502
J03WN1     A     hmm_ensemble  0.354634  0.318512  0.390732  0.737605       0.854175          500            320    141561        0.361004
J03WN1     B        tcn_clean  0.325220  0.289084  0.359348  0.675620       0.830003          500            297    141561        0.328907
J03WN1     B        consensus  0.303759  0.270175  0.335840  0.707903       0.850077          500            298    141561        0.302970
J03WPY     A baseline_nearest  0.109366  0.100271  0.119091  0.679731       0.853077          421           2088    146211        0.299925
J03WPY     A     hmm_physical  0.396067  0.358343  0.436573  0.776446       0.894671          421            291    146211        0.351770
J03WPY     A   hmm_bundesliga  0.391185  0.350937  0.431433  0.739411       0.872531          421            305    146211        0.360221
J03WPY     A     hmm_ensemble  0.375000  0.338806  0.414444  0.770157       0.889054          421            299    146211        0.338395
J03WPY     B        tcn_clean  0.360465  0.323808  0.400000  0.712408       0.868921          421            267    146211        0.358796
J03WPY     B        consensus  0.349714  0.314857  0.391457  0.746487       0.885798          421            279    146211        0.317073

## Pooled

 match panel            model  spell_f1  spell_lo  spell_hi  macro_f1  alive_dead_f1  n_gt_spells  n_pred_spells  n_graded  transitions_25
POOLED     A baseline_nearest  0.124364  0.116829  0.130691  0.672656       0.838251          921           4074    287772        0.330529
POOLED     A     hmm_physical  0.383615  0.355514  0.410923  0.756706       0.875039          921            617    287772        0.361483
POOLED     A   hmm_bundesliga  0.377320  0.347423  0.403608  0.722436       0.850964          921            631    287772        0.363542
POOLED     A     hmm_ensemble  0.364156  0.338403  0.387052  0.757172       0.872992          921            619    287772        0.350358
POOLED     B        tcn_clean  0.341549  0.314047  0.368242  0.696573       0.850725          921            564    287772        0.342919
POOLED     B        consensus  0.325234  0.299065  0.349012  0.730172       0.869358          921            577    287772        0.309623

## Paired spell-F1 differences

 match        model_a      model_b  difference       low      high  fraction_a_better
J03WN1      consensus    tcn_clean   -0.021460 -0.038526 -0.005454              0.020
J03WN1      consensus hmm_ensemble   -0.050875 -0.079508 -0.025444              0.000
J03WN1      tcn_clean hmm_ensemble   -0.029415 -0.058555 -0.002683              0.042
J03WN1 hmm_bundesliga hmm_physical   -0.007748 -0.026199  0.012131              0.260
J03WPY      consensus    tcn_clean   -0.010751 -0.027565  0.004541              0.110
J03WPY      consensus hmm_ensemble   -0.025286 -0.049379  0.000171              0.054
J03WPY      tcn_clean hmm_ensemble   -0.014535 -0.041895  0.015758              0.224
J03WPY hmm_bundesliga hmm_physical   -0.004883 -0.022361  0.011603              0.258
POOLED hmm_bundesliga hmm_physical   -0.006295 -0.018107  0.007826              0.234
POOLED      consensus hmm_ensemble   -0.038922 -0.056157 -0.020867              0.002
POOLED      tcn_clean hmm_ensemble   -0.022607 -0.043329 -0.000642              0.046
POOLED      consensus    tcn_clean   -0.016315 -0.027146 -0.004331              0.006
