# Patient Report: sub-0498

**Generated**: 2026-07-19T11:20:23.988826

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 201.320
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[FUNCTIONAL_CONNECTOME]** Elevated Default Mode within-network connectivity (z=1.73, 96th percentile, MODERATE effect)
2. **[FUNCTIONAL_CONNECTOME]** Severe decoupling between Dorsal Attention and Salience/Ventral Attention networks (z=-2.58, <1st percentile, LARGE effect)
3. **[FUNCTIONAL_CONNECTOME]** Elevated Frontoparietal control - Default mode coupling (z=1.71, 96th percentile, MODERATE effect)

## Clinical Summary
The participant exhibits a complex functional connectome profile. While robust within-network connectivity in the Default Mode and Dorsal Attention networks, alongside elevated Frontoparietal-DMN coupling, suggests a strong foundation for abstract reasoning, these strengths are partially offset by a severe decoupling between attentional networks. This suggests high cognitive potential constrained by specific inefficiencies in attentional switching, resulting in a predicted IST 2000-R total score slightly above the population mean.

## Reasoning Chain
1. Step 1: Analyzed functional connectome features for markers of cognitive efficiency.
2. Step 2: Identified convergent evidence of high-level integration (DMN/DAN integrity and FPC-DMN coupling) which typically correlates with higher IST scores.
3. Step 3: Identified significant divergent evidence (Dorsal Attention - Salience/Ventral Attention decoupling) suggesting a bottleneck in attentional resource allocation.
4. Step 4: Synthesized these opposing signals using a linear transformation of the mean z-score (0.331) relative to the population mean (200) and standard deviation (40).
5. Step 5: Concluded that the net effect of these connectivity patterns results in a score slightly above the population mean.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 37,854
- **Domains Processed**: FUNCTIONAL_CONNECTOME