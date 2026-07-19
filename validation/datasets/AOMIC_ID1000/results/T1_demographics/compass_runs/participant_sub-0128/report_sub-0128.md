# Patient Report: sub-0128

**Generated**: 2026-07-18T17:12:20.930048

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 213.240
- **Probability / Root Confidence**: 45.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is elevated (z=0.953, 83rd percentile, SMALL effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Age is lower than cohort mean (z=-1.084, 14th percentile, SMALL effect)

## Clinical Summary
The participant's predicted IST 2000-R total score is 213.24, which is slightly above the cohort mean of 200. This estimate is derived from demographic proxies, specifically education level, as no direct cognitive testing data was available. The prediction remains conservative due to the lack of direct psychometric evidence and the reliance on distal socioeconomic correlates.

## Reasoning Chain
1. Step 1: Analyzed demographic and physical features as proxies for cognitive performance due to lack of direct psychometric data.
2. Step 2: Identified education level (z=0.953) as the most significant positive predictor for intelligence in this feature set.
3. Step 3: Applied a linear weighted model to integrate demographic z-scores, anchoring the estimate to the population mean of 200.
4. Step 4: Calculated a predicted IST total score of 213.24, reflecting the slight positive bias provided by the education level and socioeconomic indicators.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 19,511
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL