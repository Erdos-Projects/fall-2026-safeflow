# Data inventory

See [data/README.md](data/README.md) for acquisition, schema and folder layout. The current study uses D12 2023-2025 measurements and the 2021-05-18 metadata snapshot, not D7. There are approximately 4.82 million cached development station-time rows, including unknown future labels, and 966,000 holdout rows. Do not interpret those counts as the number of eligible scored examples.

The holdout includes 636,806 known-outcome rows, 556,199 currently uncongested measured origins eligible for primary AP, and 3,497 onset events. The study's data-quality and missing-day checks are recorded in [verification.json](results/final/holdout/verification.json). Large raw datasets and working caches remain local.
