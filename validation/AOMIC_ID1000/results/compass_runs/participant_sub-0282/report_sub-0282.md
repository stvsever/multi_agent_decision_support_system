# Patient Report: sub-0282

**Generated**: 2026-07-18T16:11:07.850006

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 185.500
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** Education level is significantly below cohort mean (z=-2.11, 2nd percentile, LARGE effect).
2. **[PSYCHOMETRIC_PROFILES]** Openness to Experience is reduced (z=-1.09, 14th percentile, MODERATE effect).
3. **[PSYCHOMETRIC_PROFILES]** Trait anxiety is elevated (z=0.84, 80th percentile, MODERATE effect).

## Clinical Summary
Participant sub-0282 exhibits a profile characterized by significant negative deviations in educational attainment and Openness to Experience. These factors, combined with elevated trait anxiety, suggest a cognitive performance profile below the healthy young-adult cohort mean. The estimate of 185.5 reflects these structural and affective constraints on general intelligence (g).

## Reasoning Chain
1. Step 1: Analyzed demographic and psychometric features as proxies for cognitive ability in the absence of direct IST subscale data.
2. Step 2: Identified education level (z=-2.11) as the most significant structural constraint on predicted intelligence.
3. Step 3: Integrated personality markers, specifically low Openness (z=-1.09) and elevated trait anxiety (z=0.84), which collectively suggest lower cognitive exploration and higher affective interference.
4. Step 4: Applied a linear transformation model to aggregate z-scores, adjusted for the identified negative predictors, resulting in an estimate below the cohort mean of 200.
5. Step 5: Finalized the estimate at 185.5, reflecting the convergence of negative demographic and psychometric indicators.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 41,054
- **Domains Processed**: PSYCHOMETRIC_PROFILES, DEMOGRAPHICS_AND_PHYSICAL, IDENTITY_AND_BELIEF