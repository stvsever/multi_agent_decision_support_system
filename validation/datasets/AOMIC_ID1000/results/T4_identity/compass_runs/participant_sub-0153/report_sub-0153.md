# Patient Report: sub-0153

**Generated**: 2026-07-19T11:12:11.322655

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 210.560
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** High background socio-economic status (z=1.67, 95th percentile, LARGE effect)
2. **[PSYCHOLOGICAL_PROFILES]** Low BAS fun-seeking (z=-1.77, 4th percentile, LARGE effect)
3. **[PSYCHOLOGICAL_PROFILES]** Low Openness (z=-1.25, 11th percentile, MODERATE effect)

## Clinical Summary
The participant exhibits a cognitive profile estimated to be slightly above the population mean (IST 2000-R total score of 210.56). This estimate is supported by high socioeconomic and educational advantages, which are partially offset by a personality profile characterized by lower openness, reduced extraversion, and diminished behavioral activation. The prediction remains within the expected range for a healthy young-adult cohort.

## Reasoning Chain
1. Step 1: Analyzed demographic and socioeconomic markers, identifying high SES (z=1.67) as a primary positive anchor for intelligence estimation.
2. Step 2: Evaluated psychological profile, noting a cluster of negative deviations in Openness (z=-1.25), Extraversion (z=-1.17), and BAS fun-seeking (z=-1.77).
3. Step 3: Integrated these conflicting signals; while high SES supports above-average performance, the personality profile suggests lower-than-average exploratory drive and cognitive engagement.
4. Step 4: Applied linear transformation model based on cohort norms (Mean=200, SD=40) adjusted for the net influence of these features.
5. Step 5: Synthesized final estimate of 210.56, reflecting a slight positive deviation from the population mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 35,294
- **Domains Processed**: PSYCHOLOGICAL_PROFILES, DEMOGRAPHICS_AND_ANTHROPOMETRICS, IDENTITY_AND_BELIEFS