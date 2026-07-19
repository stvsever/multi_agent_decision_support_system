# Patient Report: sub-0120

**Generated**: 2026-07-19T11:09:42.002058

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 209.840
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Elevated age (z=1.55, 94th percentile, MODERATE effect) and high education level (z=0.95, 83rd percentile, SMALL effect)
2. **[PSYCHOLOGICAL_PROFILES]** Low trait anxiety (z=-0.92, 18th percentile, SMALL effect) and low BAS reward responsiveness (z=-1.07, 14th percentile, MODERATE effect)

## Clinical Summary
The participant exhibits a demographic profile (high education, high SES) that is positively associated with standardized cognitive performance. While psychological traits show minor deviations, they lack clinical significance regarding cognitive function. The predicted IST-2000R score of 209.84 places the individual slightly above the cohort mean, consistent with their favorable socioeconomic background.

## Reasoning Chain
1. Step 1: Identified demographic and socioeconomic factors as the primary predictors for IST-2000R total intelligence in the absence of direct cognitive testing.
2. Step 2: Aggregated positive z-scores from education level (z=0.95) and socioeconomic status (z=0.84) to establish an upward baseline shift from the population mean.
3. Step 3: Evaluated psychological profiles; found low-magnitude deviations (e.g., trait anxiety z=-0.92) that do not correlate with cognitive deficit, thus regularizing these inputs to avoid false-negative bias.
4. Step 4: Applied a regression model mapping the mean feature z-score (0.246) to the IST-2000R distribution (mean 200, SD 40), resulting in a predicted score of 209.84.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 30,906
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS, PSYCHOLOGICAL_PROFILES