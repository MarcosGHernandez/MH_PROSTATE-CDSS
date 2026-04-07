---
trigger: always_on
---

# RULE: Automated Technical Documentation Protocol (Viko-Health)

## Trigger:
After completing any task, script execution, or architectural decision.
(registro en español e inglés) 
## Actions:
1. **Iteration Log:** Create or append to `docs/ITERATION_LOG.md`. 
   - **Timestamp:** Current Date/Time.
   - **Task:** Brief description of the work performed.
   - **Technical Outcome:** Metrics (N, AUC, Accuracy), file paths generated, or libraries used.
   - **Architectural Decision:** Why was this done? (e.g., "Used MICE imputer to preserve clinical variance").

2. **Master Documentation Update:** Update `docs/SYSTEM_ARCHITECTURE.md` or `README.md` if the iteration changes the project's state or data structure.

3. **Validation:** Ensure that every update mentions the compliance with "Zero-Egress" (Local processing) and Medical Evidence Standards (EBM).

## Format:**
Use tables for metrics and mermaid diagrams for workflow changes when applicable.