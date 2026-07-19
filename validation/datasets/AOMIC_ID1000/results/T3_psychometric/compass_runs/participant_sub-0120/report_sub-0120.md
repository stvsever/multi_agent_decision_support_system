# Patient Report: sub-0120

**Generated**: 2026-07-18T17:15:25.019925

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 211.200
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Elevated educational level (z=0.95, 83rd percentile, MODERATE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Elevated background socio-economic status (z=0.84, 80th percentile, MODERATE effect)
3. **[MOTIVATION_AND_AFFECT]** Low trait anxiety (z=-0.92, 18th percentile, MODERATE effect)

## Clinical Summary
The participant exhibits a cognitive profile consistent with slightly above-average intelligence, estimated at 211.2 on the IST 2000-R scale. This estimate is supported by favorable socioeconomic and educational indicators, alongside a low-anxiety affective profile that likely supports efficient test performance. The absence of significant personality deviations suggests a stable cognitive baseline.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic proxies, which show consistent positive deviations (z ~ 0.8-1.0).
2. Step 2: Evaluated personality and affective traits; found them largely at baseline, with the exception of low trait anxiety, which is a positive modulator for cognitive performance.
3. Step 3: Integrated these features using a regression model anchored to the cohort mean (200) and standard deviation (40).
4. Step 4: Calculated the predicted IST total score based on the weighted contribution of socioeconomic advantages and affective stability, resulting in a score of 211.2.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 31,234
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL, PERSONALITY, MOTIVATION_AND_AFFECT