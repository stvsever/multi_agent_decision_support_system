# Patient Report: sub-0153

**Generated**: 2026-07-18T17:25:37.449153

## Prediction
- **Prediction Type**: univariate_regression
- **Primary Output**: IST_intelligence_total: 216.530
- **Probability / Root Confidence**: 65.0%
- **Confidence**: MEDIUM
- **Target Label Context**: Total Intelligence (IST 2000-R composite)

## Evaluation
- **Verdict**: SATISFACTORY
- **Checklist**: 8/8 passed

## Key Findings
1. **[BRAIN_CONNECTOME]** Salience network within-network connectivity is notably elevated (z=1.659, 95th percentile, MODERATE effect).
2. **[BRAIN_CONNECTOME]** Frontoparietal control - Default mode network integration is elevated (z=1.376, 92nd percentile, MODERATE effect).
3. **[BRAIN_CONNECTOME]** Salience - Default mode network connectivity is significantly reduced (z=-1.66, 5th percentile, MODERATE effect).

## Clinical Summary
The participant exhibits a specialized connectome profile characterized by high internal network coherence and robust executive-default mode integration, which likely supports stable, high-level cognitive processing. While the decoupling of the salience network suggests potential constraints on rapid cognitive switching, the overall neural efficiency profile is consistent with an intelligence score slightly above the cohort mean.

## Reasoning Chain
1. Step 1: Analyzed connectome profile showing high internal network integrity (Salience, DMN) but reduced cross-network communication (Salience-DMN/FPC).
2. Step 2: Identified the Frontoparietal-Default Mode integration (z=1.376) as a key protective/compensatory marker for executive function.
3. Step 3: Aggregated z-scores across the connectome hierarchy, noting a mean deviation of z=0.4133 above the population mean.
4. Step 4: Applied regression model (mean=200, SD=40) to the aggregated signal, resulting in a predicted IST-2000R score of 216.53.
5. Step 5: Calibrated prediction to reflect a high-functioning, stable cognitive profile rather than extreme outlier performance.

## Execution Details
- **Iterations**: 1
- **Selected Iteration**: 1
- **Selection Reason**: Satisfactory verdict available; chose strongest satisfactory attempt (iteration 1).
- **Tokens Used**: 44,878
- **Domains Processed**: BRAIN_CONNECTOME