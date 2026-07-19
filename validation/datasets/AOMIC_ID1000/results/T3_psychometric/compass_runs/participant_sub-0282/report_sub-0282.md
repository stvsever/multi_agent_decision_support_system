# Patient Report: sub-0282

**Generated**: 2026-07-19T11:10:40.499720

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 188.500
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Education level is notably low (z=-2.11, 2nd percentile, LARGE effect)
2. **[PSYCHOLOGICAL_PROFILES]** Openness to experience is low (z=-1.09, 14th percentile, MODERATE effect)
3. **[PSYCHOLOGICAL_PROFILES]** Trait anxiety is elevated (z=0.84, 80th percentile, SMALL effect)

## Clinical Summary
The participant's predicted IST 2000-R total intelligence score is 188.5. This estimate is derived from a profile characterized by low educational attainment and low openness to experience, which are significant negative predictors of cognitive test performance. While the participant's trait anxiety is elevated, the primary downward pressure on the score stems from the demographic and personality-based proxies. The prediction remains within the normative range for the cohort, acknowledging the lack of direct cognitive performance data.

## Reasoning Chain
1. Step 1: Analyzed demographic and psychological feature deviations relative to the healthy young-adult cohort (mean=200, sd=40).
2. Step 2: Identified significant negative pressure from low educational attainment (z=-2.11) and low openness (z=-1.09).
3. Step 3: Factored in the potential for cognitive interference due to elevated trait anxiety (z=0.84).
4. Step 4: Synthesized these indicators into a regression estimate, adjusting the population mean downward to reflect the cumulative impact of these negative predictors.
5. Step 5: Final estimate calculated as 188.5, reflecting a slight shift below the population mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 25,333
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS, PSYCHOLOGICAL_PROFILES