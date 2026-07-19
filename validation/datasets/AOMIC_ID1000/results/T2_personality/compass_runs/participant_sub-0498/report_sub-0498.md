# Patient Report: sub-0498

**Generated**: 2026-07-19T11:09:03.673383

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 195.500
- **Probability / Root Confidence**: 55.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Handedness is a significant outlier (z=-2.85, LARGE effect)
2. **[PSYCHOLOGICAL_PROFILES]** Neuroticism (NEO-FFI) is elevated (z=+1.42, MODERATE effect)
3. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age is lower than cohort mean (z=-1.08, MODERATE effect)

## Clinical Summary
Participant sub-0498 exhibits a unique profile characterized by atypical handedness and elevated neuroticism. While demographic factors are generally within normal limits, the combination of these specific deviations suggests a potential for slightly lower-than-average performance on standardized cognitive tests like the IST 2000-R, primarily due to the potential for performance interference from affective traits.

## Reasoning Chain
1. Step 1: Analyzed demographic and psychological features to estimate IST 2000-R total score.
2. Step 2: Identified handedness (z=-2.85) and neuroticism (z=+1.42) as primary deviations.
3. Step 3: Considered the negative impact of high neuroticism on standardized testing performance.
4. Step 4: Integrated demographic factors (age, education) which suggest a baseline near the cohort mean.
5. Step 5: Applied a regression estimate based on the mean of normalized features, adjusted for the dampening effect of neuroticism on cognitive performance.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 31,129
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS, PSYCHOLOGICAL_PROFILES