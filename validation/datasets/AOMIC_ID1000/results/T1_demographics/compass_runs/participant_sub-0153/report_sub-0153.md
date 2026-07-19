# Patient Report: sub-0153

**Generated**: 2026-07-18T17:12:32.889873

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 226.510
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Background socio-economic status is notably elevated (z=1.67, 95th percentile, MODERATE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is elevated (z=0.95, 83rd percentile, SMALL effect)
3. **[DEMOGRAPHICS_AND_PHYSICAL]** Biological sex (z=1.04, 85th percentile, SMALL effect)

## Clinical Summary
The participant exhibits a favorable demographic and socioeconomic profile, characterized by high background socio-economic status and educational attainment. Given the absence of direct cognitive testing, the predicted IST 2000-R total score is estimated at 226.51, which is approximately 0.66 standard deviations above the cohort mean. This estimate assumes that socioeconomic advantages correlate positively with general cognitive ability (g-factor) in this healthy young-adult population.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic features as proxies for cognitive ability in the absence of direct IST subscale data.
2. Step 2: Identified background socio-economic status (z=1.67) and education level (z=0.95) as the primary drivers of the intelligence estimate.
3. Step 3: Applied a linear regression mapping using the population mean of 200 and SD of 40, weighted by the observed z-score deviations.
4. Step 4: Calculated the predicted IST total score as 226.51, reflecting the positive influence of the participant's socioeconomic and educational background.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 19,735
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL