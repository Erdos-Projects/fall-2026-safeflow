# Data sources and preparation

Raw data is kept locally and is not redistributed by this repository. Obtain the team's Caltrans PeMS District 12 station 5-minute Parquet exports for 2023, 2024 and 2025, and the accompanying station metadata. Follow the source's access and redistribution terms.

Expected layout:

```text
data/
    D12 2023/pems_parquet/station_5min/d12_text_station_5min_2023_MM_DD.parquet
    D12 2024/pems_parquet/station_5min/d12_text_station_5min_2024_MM_DD.parquet
    D12 2025/pems_parquet/station_5min/d12_text_station_5min_2025_MM_DD.parquet
    d12_text_meta_2021_05_18.txt
```

Each daily file needs `timestamp`, `station`, `lane_type`, `speed`, `flow`, `occupancy`, and `pct_observed`. The metadata provides freeway/direction, station type, postmile and ramp locations. Windows accepts `Data` and `data` as the same directory; the portable workflow explicitly uses `data/`.

The evaluated sample has 46 D12 I-5 mainline stations. Weekdays excluding federal holidays are retained. Forecast origins are 06:00-19:55; 04:00 readings support selected recent-history features. A complete future label requires all six future readings to be measured. The default research loader excludes July-December 2025; the dedicated frozen holdout evaluator explicitly loads that already-evaluated period.

The holdout contains 966,000 rows over 125 days, with 636,806 known outcomes and 556,199 primary-AP-eligible origins. November 28, 2025 is missing among expected eligible business days. No replacement readings were fabricated. Metadata predates the measurement years, and the evaluated spatial graph is frozen to the development cohort. D7 is not included in the reported comparison.

Exact source hashes and coverage are preserved in [holdout_input_provenance.json](../results/final/holdout/holdout_input_provenance.json) and [verification.json](../results/final/holdout/verification.json). Paths inside original manifests document the original machine; release commands use repository-relative paths instead.

Processed caches are optional. `python scripts/reproduce_results.py` rebuilds development features from raw inputs; `--use-cache` reuses a matching fingerprinted cache. Data must match the recorded schema and cohort. Source availability, measured/imputed readings, complete future windows and time-of-day filtering limit what can be concluded from this sample.
