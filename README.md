# pharmacausal

Naive drug–adverse-event co-occurrence in spontaneous-reporting data is confounded: sicker patients get prescribed more drugs *and* have more adverse events. This project runs constraint-based causal discovery (PC, FCI) over FDA FAERS adverse-event reports to surface candidate drug→event hypotheses that survive confounder adjustment — and, just as importantly, is explicit about which assumptions that adjustment actually requires, whether FAERS plausibly satisfies them, and what a discovered edge does and doesn't mean.

The thesis of this project isn't "run PC on some data." It's that the confounding structure, the missing-data handling, the choice between PC and FCI, and the validation strategy all matter as much as the discovery algorithm itself — and that being honest about where a real-world dataset breaks those algorithms' assumptions is part of doing this correctly, not a caveat bolted on afterward.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. ETL | [`download.py`](src/pharmacausal/download.py), [`parse.py`](src/pharmacausal/parse.py) | Downloads a FAERS quarterly ASCII extract, deduplicates by `caseversion`, drops retracted cases, filters child tables to canonical `primaryid`s |
| 1b. Confounder audit | [`explore.py`](src/pharmacausal/explore.py) | Inventories which confounder-relevant fields exist and how populated they are |
| 2. Feature engineering | [`features.py`](src/pharmacausal/features.py) | Builds a case-level matrix: drug exposure flags, adverse-event flags, indication flags, demographic/reporting confounders |
| 3. Causal discovery | [`discovery.py`](src/pharmacausal/discovery.py), [`bounded_pc.py`](src/pharmacausal/bounded_pc.py) | Runs PC (full graph) and FCI (candidate-edge subset) with tiered background knowledge |
| 4. Validation | [`validate.py`](src/pharmacausal/validate.py) | Cross-references discovered drug→event edges against SIDER |
| 5. Visualization | [`reports/pc_fci_sider_viz.html`](reports/pc_fci_sider_viz.html) | Side-by-side PC/FCI graphs, edges colored by SIDER status |

Run order (each step's output feeds the next):

```bash
pip install -r requirements.txt
python src/pharmacausal/download.py --year 2026 --quarter 2
python src/pharmacausal/parse.py --year 2026 --quarter 2
python src/pharmacausal/explore.py --year 2026 --quarter 2
python src/pharmacausal/features.py --year 2026 --quarter 2
python src/pharmacausal/discovery.py --year 2026 --quarter 2 --subsample 10000 --fci-subsample 5000 --alpha 0.001 --max-depth 2
python src/pharmacausal/validate.py
```

Unit tests cover the logic where a silent bug would corrupt results without ever raising — drug-name normalization, age-unit conversion, background-knowledge tier assignment, and PC edge parsing ([`tests/test_pharmacausal.py`](tests/test_pharmacausal.py)):

```bash
pytest tests/ -v
```

## 1. Data: FAERS, and what confounders it actually has

FAERS is a public, no-credentialing-required quarterly extract of spontaneous adverse-event reports (`DEMO`/`DRUG`/`REAC`/`OUTC`/`RPSR`/`THER`/`INDI` tables, `$`-delimited). This run used **2026Q2: 422,458 canonical cases** after dropping 4,217 retracted case IDs and deduplicating to the latest `caseversion` per case.

Confounder coverage (full detail in `explore.py`'s output):

| Field | Coverage | Usable? |
|---|---|---|
| Report type, reporter country, drug count | 100% | yes |
| Sex | 81% | yes |
| Age | 63% | yes, needs unit normalization |
| Indication | 93.5% have a record, but 38% of *rows* are "unknown indication" | partially — see below |
| Reporter occupation | 28% | too sparse |
| Weight | 17% | too sparse, dropped |

Two structural facts bound what this analysis can mean, independent of algorithm choice:

1. **No unexposed comparison population.** FAERS contains only reported cases — there's no denominator of patients who took a drug and *didn't* have an event. No algorithm run on this data alone can estimate a population-level causal effect; at best it can find structure among the reported-case population.
2. **Selection into the dataset is a collider.** Whether a case gets reported at all depends on both the drug (media attention, litigation, the Weber effect) and the outcome's severity. That's not a missing-covariate problem — it's a selection-bias problem, and no covariate in this file represents "was this case reported," so no amount of conditioning fixes it.

Both of these are treated as first-class limitations throughout, not footnotes.

## 2. Feature engineering

422,458 cases × 143 columns: 50 drug exposure flags, 75 adverse-event flags, 10 indication flags, plus `age_years`, `sex_female`, `n_drugs`, `n_suspect_drugs`, `reporter_us`, `report_expedited`.

Notable choices, documented in [`features.py`](src/pharmacausal/features.py):

- **Drug identity = active ingredient (`prod_ai`), not brand name**, with combination products exploded on FAERS's `\` separator and salt/ester suffixes stripped (`AMLODIPINE BESYLATE` → `AMLODIPINE`) — needed for the SIDER cross-reference in step 4 to work at all.
- **Exposure flags are role-agnostic**: primary suspect, secondary suspect, and concomitant drugs all count as "exposed." `role_cod` is the *reporter's* suspicion about which drug caused the event, not ground truth — using only "primary suspect" drugs as exposures would bake the reporter's own causal judgment into the input before discovery ever runs, which is exactly the kind of shortcut this project is trying to avoid.
- **Indication limited to the top 10 known categories**, excluding "unknown indication." Indication is a genuine confounder candidate but is also frequently a direct cause of the prescription itself — it can legitimately sit upstream of a drug node in the true graph, so it's included as a node rather than "adjusted and forgotten."

## 3. Causal discovery: PC and FCI, and what each one actually assumes

### Missing data
`age_years` (37% missing) and `sex_female` (19% missing) are the only columns with missingness. Handled with **listwise deletion** (232,233 rows, 55%, survive), not a missingness-indicator column — a missingness flag for age/sex would mostly encode *which report types bother filling in demographics*, and would likely become a spurious confounding hub entangled with `report_expedited`/`reporter_us` for reasons that have nothing to do with drugs or events.

### PC: causal sufficiency
PC assumes no unmeasured variable confounds two or more measured ones. **This assumption is false for FAERS** — indication captures some of "why was this drug given," but disease severity, comorbidity burden beyond indication, and reporting stimulus are all unmeasured and plausibly affect multiple nodes at once. PC's output should be read as *candidates under an assumption known not to hold*, not as a result to trust standalone.

### FCI: relaxing causal sufficiency (but not selection bias)
FCI allows latent common causes and represents uncertainty about them with `o->`/`<->` marks in the output PAG. This is the more honest assumption for FAERS — but **FCI does not solve the selection-bias problem** described above. Modeling selection bias soundly requires representing the selection mechanism as an explicit node, and "was this case reported" isn't a variable we have; it's the sampling frame itself. Every edge in this graph should be read as *structure among reported cases*, never as a population-level effect.

### Faithfulness
No violations found, but one mechanical near-violation is worth naming: `n_suspect_drugs ≤ n_drugs` by construction, not by any causal mechanism — a synthetic dependency from feature engineering, not the data-generating process. Kept both because they encode different things (polypharmacy vs. suspicion burden), but it's not something PC/FCI should be "discovering."

### Background knowledge: fixing orientation PC/FCI can't get from data alone
An early run oriented edges like `Headache --> ABALOPARATIDE` — a *reaction* apparently causing the *drug*. PC/FCI orient purely from conditional-independence structure; nothing in that structure encodes "drug exposure precedes a reported reaction." Fixed with **tiered background knowledge** ([`discovery.py`](src/pharmacausal/discovery.py)): demographics/report-metadata → indication/polypharmacy → drug → event, where later tiers can never be oriented as causing earlier ones. This is standard practice (equivalent to Tetrad's knowledge tiers) and is essential here, not optional polish.

### Computational tractability (a real finding, not just an implementation note)
- causal-learn's off-the-shelf `pc()` has **no cap on conditioning-set size**, which is infeasible at 141 variables. [`bounded_pc.py`](src/pharmacausal/bounded_pc.py) forks the skeleton search with a depth cap — a standard, documented completeness/tractability tradeoff.
- FCI's runtime scales badly with **both** node count (15→30→50 vars: ~1s → ~5s → ~86s at fixed N) and, more sharply, **row count** — a 48-variable PC-linked subset ran in 4.5s at N=3,000 but didn't finish in 30 minutes at N=10,000. This pushed a concrete design decision: **decouple FCI's sample size from PC's**. FCI here runs as a confirmatory check on PC's own candidate edges (drug/event nodes PC connected, plus all confounders — 64 nodes), not as an independent full-scale discovery pass, so it doesn't need PC's N to do that job.

### Results
- **PC** (N=10,000, α=0.001, depth≤2): 489 total edges, **36 drug→event edges**.
- **FCI** (N=5,000, 64-node PC-linked subset, same α/depth): 85 total edges, **22 drug→event edges**.
- **20 of 22** FCI drug-event edges match a PC edge exactly; 16 PC-only edges didn't survive the FCI subset (read with caution — FCI ran on less data and a narrower conditioning set, so "didn't survive" conflates relaxed causal sufficiency with reduced power, and the two effects aren't separated here); 2 edges appeared only in FCI.
- FCI's more interesting output isn't on the drug→event edges themselves (tier-locked to a single orientation) but on **16 other edges** it marked as likely sharing a latent common cause: symptom clusters like `Dizziness <--> Headache`, `Pruritus <--> Rash` (plausibly one reaction syndrome producing several correlated symptoms, not one symptom causing another), and co-prescription clusters like `ALBUTEROL o-> MONTELUKAST` (both asthma drugs). This is exactly the kind of structure naive co-occurrence counting would misread as direct causation.

## 4. Validation: SIDER, not DrugBank

Our discovered edges are **drug → adverse-event** associations. DrugBank's structured interaction data is **drug-drug interactions** — a different relationship type entirely; cross-referencing against it would have produced a meaningless number. [SIDER](http://sideeffects.embl.de) (Side Effect Resource) is the correct match: it's mined from drug package-insert labels into structured drug→MedDRA-side-effect pairs, the same relationship type as our discovery output.

| | Edges | SIDER coverage | Precision (covered subset) |
|---|---|---|---|
| PC | 36 | 33% (12/36) | 58% (7/12) |
| FCI | 22 | 27% (6/22) | 67% (4/6) |

**Coverage and precision are reported separately on purpose.** SIDER 4.1 was last updated in 2015; half of our candidate drugs postdate that cutoff (dupilumab, semaglutide, tirzepatide, bimekizumab, abaloparatide) or are biologics SIDER covers poorly even when older (rituximab, approved 1997, has zero SIDER entries). Scoring "no SIDER data" the same as "wrong" would understate the method and overstate SIDER as ground truth.

**Standout matches**: `ROSUVASTATIN → Myalgia` (textbook statin myalgia — a strong sanity check), `MEDROXYPROGESTERONE → Meningioma` (a real, actively-discussed pharmacovigilance signal — recovering it from FAERS alone via causal discovery is a genuinely good result), `APIXABAN → Anaemia` (mechanistically sensible: anticoagulant → bleeding-related anemia).

**A finding worth stating plainly rather than burying**: several unmatched edges trace to a scoping issue, not algorithm failure. `MONTELUKAST → Asthma` is very likely indication bleeding into the reaction field (asthma is montelukast's *indication*). And PTs like `Therapy interrupted`, `Off label use`, `Wrong technique in product usage process`, `Incorrect dose administered` aren't clinical adverse reactions at all — they're **administrative/process MedDRA terms** FAERS codes for regulatory tracking, and the top-75-by-frequency event selection in step 2 didn't filter them out. A concrete, fixable improvement for a v2: exclude a stoplist of process PTs from the event vocabulary.

## 5. Visualization

Side-by-side PC/FCI bipartite drug↔event graphs, edges colored by SIDER status (match / not labeled / no data), plus a dedicated panel for FCI's latent-confounding signals. Self-contained HTML/SVG (no framework dependency), styled to sit alongside [Causeway](../Causeway) without literally reusing its Dash/Cytoscape stack — see [`reports/pc_fci_sider_viz.html`](reports/pc_fci_sider_viz.html).

## Limitations, stated plainly

- **Selection bias into FAERS is not addressed by any method used here.** This is the single biggest caveat and it's structural, not fixable by a better algorithm on this data alone.
- **No unexposed denominator** — nothing here estimates absolute or relative risk, only structure among reported cases.
- **Causal sufficiency is violated for PC**; FCI relaxes it but the 141-variable graph was too expensive for FCI to run directly, so FCI only checked a subset of PC's candidates, not the full space.
- **The 141-variable and depth caps** are documented tractability tradeoffs, not evidence of a "complete" search — conditional independencies that only emerge from higher-order conditioning sets could be missed.
- **SIDER 2015 vintage** under-covers newer drugs and biologics, which lowers coverage without meaning anything about correctness.
- **Administrative/process MedDRA terms** contaminate the top-75-by-frequency event vocabulary and should be filtered in a v2.
- **N=10,000 (PC) / N=5,000 (FCI)** are subsamples of the 232,233-row complete-case pool, chosen for tractability given FCI's poor scaling with row count — a larger N was tried (30,000 rows) and did not converge within an hour.

## Reproducing

```
pharmacausal/
├── src/pharmacausal/       # all pipeline code
├── tests/                  # unit tests for the pure/deterministic logic
├── data/raw/, data/processed/   # gitignored — regenerate via the scripts above
├── reports/                # visualization artifact
└── requirements.txt
```

Everything downstream of `download.py` is deterministic given the same FAERS extract and `random_state=0` subsampling; the FAERS quarterly URL pattern (`faers_ascii_<year>q<quarter>.zip`) means results can be regenerated or extended to other quarters by changing `--year`/`--quarter`.

## License

[MIT](LICENSE)
