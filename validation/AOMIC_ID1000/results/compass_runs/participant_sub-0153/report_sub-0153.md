# Patient Report: sub-0153

**Generated**: 2026-07-18T16:10:43.329332

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 210.840
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[DEMOGRAPHICS_AND_PHYSICAL]** High socioeconomic status (z=1.67, 95th percentile, LARGE effect) and education level (z=0.95, 83rd percentile, MODERATE effect) provide a strong positive baseline for cognitive performance.
2. **[PSYCHOMETRIC_PROFILES]** Low Openness (z=-1.25, 11th percentile, MODERATE effect) and low Conscientiousness (z=-0.82, 21st percentile, SMALL effect) indicate reduced intellectual curiosity and self-discipline.
3. **[PSYCHOMETRIC_PROFILES]** Low extraversion (z=-1.17, 12th percentile, MODERATE effect) and low fun-seeking (z=-1.77, 4th percentile, LARGE effect) suggest a passive behavioral style.

## Clinical Summary
Participant sub-0153 exhibits a profile of high socioeconomic and educational advantage, which is strongly associated with higher cognitive performance. However, this is tempered by a personality profile characterized by low Openness and low Conscientiousness, which likely limits the participant's engagement with standardized cognitive tasks. The predicted IST 2000-R score of 210.84 reflects a balance between these high environmental assets and lower personality-driven intellectual curiosity.

## Reasoning Chain
1. Step 1: Identified strong positive predictive signal from socioeconomic and educational background (mean z=1.31).
2. Step 2: Identified significant negative predictive signal from personality traits (Openness, Conscientiousness, Extraversion) which typically correlate with lower standardized test performance.
3. Step 3: Integrated demographic advantages with personality-based constraints to adjust the population mean (200) upward.
4. Step 4: Applied regression-based estimation using the weighted influence of these domains to arrive at a final score of 210.84.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 35,007
- **Domains Processed**: PSYCHOMETRIC_PROFILES, DEMOGRAPHICS_AND_PHYSICAL, IDENTITY_AND_BELIEF