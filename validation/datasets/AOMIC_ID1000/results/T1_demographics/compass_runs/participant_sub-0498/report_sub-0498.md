# Patient Report: sub-0498

**Generated**: 2026-07-18T17:13:03.696749

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 180.940
- **Probability / Root Confidence**: 45.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Handedness deviation (z=-2.85, <1st percentile, LARGE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Age deviation (z=-1.08, 14th percentile, MODERATE effect)
3. **[DEMOGRAPHICS_AND_PHYSICAL]** Body mass index deviation (z=-0.91, 18th percentile, SMALL effect)

## Clinical Summary
The participant's predicted IST 2000-R total intelligence score is 180.94. This estimate is derived from demographic and physical attributes, specifically noting a significant negative deviation in handedness and age. Given the absence of direct psychometric testing, this score reflects a statistical inference based on available distal proxies and should be interpreted with caution as a population-level estimate rather than a clinical assessment.

## Reasoning Chain
1. Step 1: Analyzed available demographic and physical features as proxies for cognitive performance.
2. Step 2: Identified significant negative deviations in handedness (z=-2.85) and age (z=-1.08) as the primary drivers of the prediction.
3. Step 3: Discounted the aggregate domain z-score (1.133) as non-specific demographic noise, prioritizing granular leaf-level data.
4. Step 4: Applied a downward adjustment from the population mean (200) based on the weighted average of negative deviations.
5. Step 5: Calculated the final regression estimate of 180.94, reflecting a conservative shift below the cohort mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 22,415
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL