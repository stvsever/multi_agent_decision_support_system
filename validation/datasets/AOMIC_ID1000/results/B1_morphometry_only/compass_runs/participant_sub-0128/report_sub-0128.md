# Patient Report: sub-0128

**Generated**: 2026-07-18T17:23:35.188931

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 230.190
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[BRAIN_MORPHOMETRY]** Total cortical gray matter volume is elevated (z=1.53, 94th percentile, MODERATE effect)
2. **[BRAIN_MORPHOMETRY]** Left hippocampus volume is elevated (z=1.68, 95th percentile, MODERATE effect)
3. **[BRAIN_MORPHOMETRY]** Consistent positive deviation across global brain measures and cortical thickness (mean z=0.75)

## Clinical Summary
Participant sub-0128 exhibits a structural brain phenotype characterized by high-normal gray matter volumes and cortical thickness. The consistent positive deviation across global morphometric indices, particularly in cortical gray matter and hippocampal structures, is associated with superior cognitive performance in healthy young-adult cohorts. The predicted IST 2000-R total score of 230.19 places the participant in the upper quartile of the expected distribution.

## Reasoning Chain
1. Step 1: Analyzed hierarchical brain morphometry data, identifying a consistent positive bias across global gray matter and cortical thickness metrics.
2. Step 2: Evaluated the significance of specific high-z features (Total cortical gray matter z=1.53, Left hippocampus z=1.68) as established neurobiological correlates of cognitive capacity.
3. Step 3: Applied a regression model to map the aggregate structural profile (mean z=0.75) onto the IST 2000-R scale (mean 200, SD 40).
4. Step 4: Calculated the predicted IST total score as 230.19, reflecting a performance approximately 0.75 standard deviations above the cohort mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 39,478
- **Domains Processed**: BRAIN_MORPHOMETRY