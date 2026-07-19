# Patient Report: sub-0673

**Generated**: 2026-07-18T17:13:21.671135

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 221.690
- **Probability / Root Confidence**: 35.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is elevated (z=0.953, 83rd percentile, MODERATE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Background socio-economic status is elevated (z=0.837, 80th percentile, MODERATE effect)
3. **[DEMOGRAPHICS_AND_PHYSICAL]** Age is lower than cohort mean (z=-1.23, 11th percentile, MODERATE effect)

## Clinical Summary
The participant's predicted IST 2000-R total score is 221.69. This estimate is derived from demographic and socioeconomic proxies, specifically high educational attainment and favorable background socio-economic status, which are positively correlated with general cognitive ability. Given the absence of direct cognitive performance metrics, this prediction reflects a population-level association and should be interpreted with caution, as individual cognitive variance remains unmeasured.

## Reasoning Chain
1. Step 1: Identified the target as a regression task for IST 2000-R (Mean=200, SD=40).
2. Step 2: Evaluated available features; cognitive and educational performance data are absent, necessitating reliance on demographic proxies.
3. Step 3: Weighted socioeconomic indicators (Education level z=0.953, SES z=0.837) as the primary positive predictors for cognitive ability.
4. Step 4: Accounted for age (z=-1.23) and sex (z=1.042) as secondary demographic covariates.
5. Step 5: Applied a shrinkage-adjusted regression estimate, shifting the population mean (200) upward based on the positive socioeconomic signal, resulting in a predicted score of 221.69.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 27,550
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL