# Patient Report: sub-0498

**Generated**: 2026-07-18T17:14:48.984379

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 191.910
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Handedness deviation (z=-2.85, LARGE effect)
2. **[DEMOGRAPHICS_AND_PHYSICAL]** Age (z=-1.08, MODERATE effect)
3. **[PERSONALITY]** Neuroticism (z=+1.42, MODERATE effect)

## Clinical Summary
The participant exhibits a cognitive profile estimated to be slightly below the cohort mean of 200. The prediction is driven by consistent negative deviations in demographic and socioeconomic markers, specifically age and handedness, which outweigh the minor positive signals from personality traits. The estimate of 191.9 reflects a conservative adjustment given the absence of direct cognitive testing data.

## Reasoning Chain
1. Step 1: Analyzed demographic and personality features as proxies for cognitive performance.
2. Step 2: Identified consistent negative deviations across physical and socioeconomic leaf nodes (age, handedness, BMI, education).
3. Step 3: Evaluated the conflict between positive root-level aggregates and negative leaf-level data; prioritized granular leaf data as per clinical guidance.
4. Step 4: Applied a regression model mapping the mean z-score of -0.202 to the IST-2000R scale (mean 200, SD 40).
5. Step 5: Calculated the final estimate of 191.9, reflecting a mild downward adjustment from the population mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 24,522
- **Domains Processed**: DEMOGRAPHICS_AND_PHYSICAL, PERSONALITY