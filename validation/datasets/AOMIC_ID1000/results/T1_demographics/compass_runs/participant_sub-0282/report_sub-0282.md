# Patient Report: sub-0282

**Generated**: 2026-07-18T17:12:50.669998

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 184.040
- **Probability / Root Confidence**: 45.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is significantly below cohort mean (z=-2.11, 2nd percentile, LARGE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Body mass index is below cohort mean (z=-1.57, 6th percentile, MODERATE effect)
3. **[DEMOGRAPHICS_AND_PHYSICAL]** Age is below cohort mean (z=-1.67, 5th percentile, MODERATE effect)

## Clinical Summary
The participant's predicted IST 2000-R total intelligence score is 184.04. This estimate is derived from demographic and physical proxies, specifically noting a significantly lower education level and lower BMI compared to the healthy young-adult cohort. Given the absence of direct cognitive performance data, this prediction reflects a conservative downward adjustment from the population mean of 200, acknowledging the limitations of using demographic markers as surrogates for cognitive ability.

## Reasoning Chain
1. Step 1: Identified that no direct cognitive test scores are available for the IST 2000-R composite.
2. Step 2: Analyzed demographic and physical features as proxies for general intelligence (g-factor).
3. Step 3: Noted consistent negative deviations in education level (z=-2.11), BMI (z=-1.57), and age (z=-1.67).
4. Step 4: Evaluated the hierarchical conflict where the aggregate domain score (z=1.124) was positive, but granular leaf nodes were negative; prioritized leaf nodes as more reliable indicators.
5. Step 5: Applied a linear mapping from the mean z-score of relevant features to the IST 2000-R scale (Mean 200, SD 40).

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 28,272
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL