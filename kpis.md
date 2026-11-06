# Prediction and warning KPIs

A warning policy is an alert rule: issue an alert when a model probability is at least its frozen threshold. A warning budget is an illustrative acceptable-burden target used to choose that threshold on calibration data. It is not money and is not guaranteed on future data. Stakeholders have not validated these particular limits as operating requirements.

| Measure | Definition | Purpose |
| --- | --- | --- |
| Primary AP | Average precision on measured, currently uncongested origins with known next-six-reading outcomes | Probability ranking |
| Brier score | Mean squared probability error on all known future targets | Probability quality |
| Early onset catch | Fraction of onsets warned 15-30 minutes before they start | Advance detection |
| Wrong-alert duration | Five minutes per known-negative alert reading, summed across stations per day | Duration-policy calibration target: at most 4 hours/day |
| False-warning episodes | Starts of consecutive known-negative alert readings at the same station, with exact 5-minute adjacency | Episode-policy calibration target: at most 20/day |

An onset is measured speed below 40 mph following 30 measured uncongested minutes. The prediction target is any measured speed below 40 mph in the next six readings, not sustained congestion and not a crash or incident label.

One continuous 30-minute wrong warning is one episode and 0.5 hours; six separated five-minute wrong warnings are six episodes and 0.5 hours. Duration and episode targets are separate policies. The duration policy is the primary warning comparison; the episode policy and fixed threshold 0.5 are secondary. Each policy maximizes early detection within its calibration target, with thresholds frozen before holdout access. Circles/squares/triangles denote duration/episode/0.5 in the original tradeoff figure; color denotes model.

All three duration policies exceed their 4-hour/day target on holdout; all episode policies exceed 20/day. Budget non-compliance is a study finding, not something corrected by post-holdout retuning. AP and early-catch differences do not establish superior operation at identical realized cost. Further improvements require new independent validation.
