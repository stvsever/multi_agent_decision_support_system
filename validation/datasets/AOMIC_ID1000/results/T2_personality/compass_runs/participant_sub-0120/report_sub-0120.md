# Patient Report: sub-0120

**Generated**: 2026-07-19T11:07:50.643091

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 222.550
- **Probability / Root Confidence**: 45.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age (z=1.55, 94th percentile, MODERATE effect) and Education level (z=0.95, 83rd percentile, SMALL effect) indicate favorable environmental factors for cognitive performance.
2. **[PSYCHOLOGICAL_PROFILES]** Conscientiousness (z=-0.65, 26th percentile, SMALL effect) is slightly below average, which may marginally offset positive demographic indicators.

## Clinical Summary
Participant sub-0120 exhibits a demographic profile (higher age and education level) that correlates with slightly above-average cognitive performance in healthy young-adult cohorts. Personality traits are largely unremarkable, with the exception of slightly lower conscientiousness. The predicted IST-2000-R total score of 222.55 is consistent with a high-functioning individual within the normal distribution, though the lack of direct cognitive performance data necessitates a conservative interpretation.

## Reasoning Chain
1. Step 1: Analyzed demographic and psychological feature profiles for sub-0120.
2. Step 2: Identified that demographic features (age, education, SES) show consistent positive deviations (z > 0.8), while personality traits are largely within the normal range.
3. Step 3: Applied a regression-based approach using the mean z-score of prioritized demographic features to estimate the IST-2000-R total score.
4. Step 4: Calculated the predicted value by adjusting the cohort mean (200) by the weighted influence of the observed demographic deviations.
5. Step 5: Finalized the estimate at 222.55, reflecting a moderate positive shift from the population mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 23,877
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS, PSYCHOLOGICAL_PROFILES