# Patient Report: sub-0498

**Generated**: 2026-07-19T11:07:20.367192

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 189.130
- **Probability / Root Confidence**: 35.0%
- **Confidence**: LOW
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Handedness deviation (z=-2.85, LARGE effect)
2. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Age (z=-1.08, MODERATE effect)
3. **[DEMOGRAPHICS_AND_ANTHROPOMETRICS]** Education and SES (z=-0.58, -0.42, SMALL effect)

## Clinical Summary
The participant presents with a demographic profile characterized by consistent, mild-to-moderate negative deviations from the cohort mean across age, education, and socioeconomic indicators. In the absence of direct cognitive testing, these proxies suggest a total intelligence score slightly below the population average of 200. The prediction is constrained by the lack of specific psychometric data.

## Reasoning Chain
1. Step 1: Analyzed demographic and anthropometric features as proxies for cognitive performance.
2. Step 2: Observed consistent negative deviations across all available features (age, education, SES, BMI, handedness).
3. Step 3: Calculated an estimated score based on the mean z-score of -0.27 relative to the population mean of 200 and SD of 40.
4. Step 4: Applied a shrinkage adjustment to account for the lack of direct cognitive psychometric data, centering the prediction slightly below the population mean to reflect the negative demographic profile.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 24,782
- **Domains Processed**: DEMOGRAPHICS_AND_ANTHROPOMETRICS