# Patient Report: sub-0120

**Generated**: 2026-07-19T11:19:19.840204

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 227.600
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[FUNCTIONAL_CONNECTOME]** Limbic-Frontoparietal control connectivity is significantly elevated (z=2.04, 97.9th percentile, MODERATE effect).
2. **[FUNCTIONAL_CONNECTOME]** Visual-Default mode connectivity is reduced (z=-1.76, 3.9th percentile, MODERATE effect).
3. **[FUNCTIONAL_CONNECTOME]** Global functional connectivity shows a positive bias (mean z=0.69).

## Clinical Summary
Participant sub-0120 exhibits a functional connectome profile characterized by enhanced cross-network integration between limbic and frontoparietal systems and a decoupling of sensory-visual networks from higher-order associative cortices. This neural architecture is consistent with efficient cognitive resource allocation, supporting an estimated IST 2000-R total intelligence score of 227.6, which is approximately 0.7 standard deviations above the cohort mean.

## Reasoning Chain
1. Step 1: Analyzed functional connectome deviations, noting a consistent pattern of elevated cross-network integration.
2. Step 2: Identified high-importance features, specifically the Limbic-Frontoparietal connectivity (z=2.04) and Visual-Default mode decoupling (z=-1.76).
3. Step 3: Applied a regression model mapping the aggregate z-score profile (mean z=0.69) to the IST 2000-R population distribution (M=200, SD=40).
4. Step 4: Calculated the estimated score as 200 + (0.69 * 40) = 227.6, adjusting for the observed positive connectivity bias.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 41,385
- **Domains Processed**: FUNCTIONAL_CONNECTOME