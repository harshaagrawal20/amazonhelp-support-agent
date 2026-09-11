# Golden Evaluation Set: Inter-Annotator Agreement Report

> **STATUS**: **PENDING DUAL ANNOTATION** (Methodological integrity enforced: no synthetic agreement scores).

- **Total Examples**: `200`
- **Annotator 1 Progress**: `16 / 200`
- **Annotator 2 Progress**: `0 / 200`
- **Dual Annotated Overlap**: `0 / 200`

### Agreement Evaluation Instructions

1. Annotator 1 completes labeling via `python scripts/annotate_golden.py --annotator annotator_1`.
2. Annotator 2 independently completes labeling via `python scripts/annotate_golden.py --annotator annotator_2`.
3. Run `python scripts/analyze_annotation_agreement.py` to calculate observed agreement and Cohen's Kappa.