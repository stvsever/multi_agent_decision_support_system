# Patient Report: sub-0282

**Generated**: 2026-07-19T11:07:06.243519

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 194.760
- **Probability / Root Confidence**: 35.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Education level is significantly below cohort mean (z=-2.11, 2nd percentile, LARGE effect)
2. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age is below cohort mean (z=-1.67, 5th percentile, MODERATE effect)
3. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Body mass index is below cohort mean (z=-1.57, 6th percentile, MODERATE effect)

## Clinical Summary
The participant's estimated IST-2000R total score is 194.76. This estimate is derived from demographic and socioeconomic proxies, as no direct cognitive performance data was available. The prediction is heavily influenced by the participant's low educational level relative to the cohort, which acts as a negative anchor for general intelligence estimation, partially offset by other demographic variables.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic features as proxies for general intelligence (IST-2000R).
2. Step 2: Identified significant negative deviation in education level (z=-2.11) and age (z=-1.67).
3. Step 3: Balanced these against the positive deviation in biological sex (z=1.04) and neutral socioeconomic status (z=0.0).
4. Step 4: Applied population parameters (mean=200, SD=40) to the aggregated z-score profile.
5. Step 5: Calculated an estimated score of 194.76, reflecting the downward pressure of low educational attainment on the composite intelligence estimate.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 19,719
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS