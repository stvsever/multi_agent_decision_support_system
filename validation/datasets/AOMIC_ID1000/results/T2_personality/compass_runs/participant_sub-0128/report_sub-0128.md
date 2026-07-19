# Patient Report: sub-0128

**Generated**: 2026-07-18T17:13:55.130863

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 215.110
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[PERSONALITY]** Extraversion (z=1.64, 95th percentile, MODERATE effect) and low Neuroticism (z=-1.07, 14th percentile, SMALL effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level (z=0.95, 83rd percentile, SMALL effect)

## Clinical Summary
Participant sub-0128 exhibits a personality and demographic profile consistent with slightly above-average cognitive performance. The combination of high extraversion, high conscientiousness, and high educational attainment suggests a robust cognitive reserve. The predicted IST 2000-R total score is 215.11, which is approximately 0.38 standard deviations above the cohort mean.

## Reasoning Chain
1. Step 1: Analyzed demographic and personality features as proxies for cognitive ability in the absence of direct IST subscale scores.
2. Step 2: Identified positive predictive signals in Extraversion (z=1.64), Conscientiousness (z=0.55), and Education level (z=0.95).
3. Step 3: Aggregated these signals to estimate a deviation from the cohort mean (200).
4. Step 4: Applied a regression model to map the aggregate z-score (0.378) to the IST scale, resulting in a predicted score of 215.11.
5. Step 5: Validated against the population SD of 40, ensuring the prediction remains within the plausible range (68-296).

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 29,330
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL, PERSONALITY