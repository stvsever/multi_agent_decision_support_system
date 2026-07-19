# Patient Report: sub-0673

**Generated**: 2026-07-19T11:07:35.779773

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
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Education level is elevated (z=0.953, 83rd percentile, SMALL effect)
2. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Socio-economic status is elevated (z=0.837, 80th percentile, SMALL effect)
3. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age is reduced (z=-1.23, 11th percentile, MODERATE effect)

## Clinical Summary
The participant's predicted IST 2000-R total intelligence score is 212.5. This estimate is derived from demographic and socioeconomic proxies, specifically high educational attainment and favorable socioeconomic background, which correlate with higher cognitive performance. Given the absence of direct psychometric data, this prediction remains an estimate with significant uncertainty, anchored toward the cohort mean of 200.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic features as proxies for cognitive performance in the absence of direct IST subscale data.
2. Step 2: Identified education level (z=0.953) and socioeconomic status (z=0.837) as the primary positive indicators for intelligence.
3. Step 3: Evaluated biological characteristics (age, sex, BMI) as non-informative noise for cognitive performance estimation.
4. Step 4: Applied a regression model anchored to the population mean of 200, adjusting upward based on the combined positive socioeconomic and educational signal.
5. Step 5: Final prediction calculated as 212.5, reflecting a modest deviation above the mean consistent with the observed demographic profile.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 22,220
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS