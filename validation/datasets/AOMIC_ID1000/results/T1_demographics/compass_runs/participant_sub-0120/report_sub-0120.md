# Patient Report: sub-0120

**Generated**: 2026-07-19T11:06:25.390171

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 212.500
- **Probability / Root Confidence**: 35.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age is elevated (z=1.55, 94th percentile, MODERATE effect)
2. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Education level is elevated (z=0.95, 83rd percentile, SMALL effect)
3. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Background socio-economic status is elevated (z=0.84, 80th percentile, SMALL effect)

## Clinical Summary
The participant exhibits a favorable demographic profile characterized by higher-than-average educational attainment and socioeconomic background relative to the cohort. While these factors are positively correlated with cognitive performance, they are distal proxies and do not constitute direct evidence of intelligence. The estimated IST 2000-R score is adjusted moderately above the population mean of 200, reflecting these demographic advantages while accounting for the significant uncertainty inherent in the absence of direct psychometric data.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic inputs which show a consistent positive deviation from the population mean.
2. Step 2: Acknowledged the absence of direct cognitive, psychometric, or executive function data, which limits the predictive precision.
3. Step 3: Applied a shrinkage factor to the raw demographic z-score correlations to prevent over-estimation, as these are distal proxies for intelligence.
4. Step 4: Calculated a conservative estimate by regressing the demographic-derived signal toward the population mean of 200, resulting in a moderate upward adjustment.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 22,367
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS