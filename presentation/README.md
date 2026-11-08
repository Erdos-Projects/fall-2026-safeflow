# Presentation materials

The team's slide deck is available as a [PDF](Final_Presentation.pdf) and an [editable PowerPoint](Final_Presentation.pptx).

[summary.md](summary.md) is the executive summary, also available as a [PDF](summary.pdf). This is a comparative study: all three models are retained and neither boosted trees nor Transformer is designated the final model. The explicit decision is that both leading candidates support predictive feasibility while neither meets the holdout warning-budget target.

Use figures in [results/final/figures](../results/final/figures) and numeric tables under [results/final](../results/final/README.md). The showcase notebook regenerates these figures and verifies holdout scores. Keep development averages, one-fold benchmark scores and holdout results labeled separately.

Suggested narrative: problem and stakeholders; data and EDA; features kept/dropped/added; timetable and model comparisons; holdout performance and actual warning burden; feasibility, limitations and future independently evaluated improvements. Explain warning policy and budget before showing the tradeoff plots.
