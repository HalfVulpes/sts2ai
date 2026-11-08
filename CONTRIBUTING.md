# Contributing

Ideas, alternative policies, curriculum samplers and failure analyses are welcome.
Contributions to original project code are accepted under GPL-2.0-only. Keep any
third-party notices and identify imported code before including it.

For a reproducible result, include the game version and assembly hash, code revision,
checkpoint hash, seed protocol and seed list, character, ascension, search budget,
observation contract, failures/timeouts and evaluation count. Preserve unsuccessful
seeds and training runs. Separate development tuning from final evaluation.

Run `python3 -m pytest -q tests` for source changes. Changes to the engine, encoder
or driver also need a compatible installed game and an integration check. Do not
commit game DLLs, extracted resources, decompiled files, credentials, private runtime
archives, machine-specific checkpoints, weights, logs or other binary files: binaries go to the
GitHub release and records to the records bundle.

The primary policy evaluation uses greedy zero-search inference. Search experiments
are welcome with separate labels and measured budgets. New algorithm claims should
compare matched controls and independent training initializations; more game seeds
cannot substitute for more independent training runs.
