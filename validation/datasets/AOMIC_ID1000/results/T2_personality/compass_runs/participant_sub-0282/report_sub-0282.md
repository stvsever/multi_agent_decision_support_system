# Patient Report: sub-0282

**Generated**: 2026-07-18T17:14:33.998346

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 190.920
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is significantly below cohort mean (z=-2.11, 2nd percentile, LARGE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Body mass index is low (z=-1.58, 6th percentile, MODERATE effect)
3. **[PERSONALITY]** Openness to experience is low (z=-1.09, 14th percentile, MODERATE effect)

## Clinical Summary
The participant's predicted IST 2000-R score is 190.92, which is slightly below the population mean of 200. This estimate is derived from demographic and personality proxies, specifically a notably low education level and reduced openness to experience. The prediction is tempered by the lack of direct cognitive performance data, resulting in a moderate confidence level.

## Reasoning Chain
1. Step 1: Establish baseline population mean (200) and standard deviation (40) for the IST 2000-R.
2. Step 2: Identify primary negative signals: Education level (z=-2.11) and BMI (z=-1.58) suggest potential environmental or developmental limitations.
3. Step 3: Integrate personality markers: Low Openness (z=-1.09) correlates with lower intellectual curiosity, providing a secondary negative weight.
4. Step 4: Apply regression model: Given the absence of direct cognitive data, the prediction is anchored to the mean and adjusted downward by approximately 0.25 standard deviations based on the aggregate negative signal of the identified proxies.
5. Step 5: Final calculation: 200 - (0.25 * 40) = 190.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 24,149
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL, PERSONALITY