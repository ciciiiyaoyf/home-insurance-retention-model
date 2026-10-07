# Source schema and derived fields

This directory documents a generalized contract for the retention pipeline. It
contains column definitions and code-derived metadata, with no policy records,
customer values, observed distributions, or original warehouse connection details.
Field names that the modeling code references remain stable so the contract can
be checked against that code.

| Artifact | Purpose |
| --- | --- |
| `source_columns.csv` | All raw source fields, including disabled predictors and required build inputs. |
| `model_derived_columns.csv` | Outputs calculated by the retention build, including labels, diagnostics, and compatibility fields. |
| `manifest.json` | Coverage counts, optional-output distinction, and artifact hashes. |
| `../../b_data_acquisition/sql/create_policy_exposure_history.sql` | Non-destructive Snowflake source-table declaration. |
| `generate_schema.py` | Regenerates this contract from the configuration and declared Python constants without importing the application or connecting to a warehouse. |

## Coverage and row grain

The raw table is
`INSURANCE_ANALYTICS.RETENTION.POLICY_EXPOSURE_HISTORY`. Each row represents an
exposure interval within a policy contract and a scheduled policy term. A contract
can have multiple intervals within a term and multiple successive terms. The
retention build later produces one eligible row per contract and anchor-term
start; the raw source is not already at that grain.

The 404-row configuration contains 390 raw source fields and 14 fields that the
build recomputes. Three additional build inputs bring the raw source to **393
columns**: `EARNED_EXPOSURE_VALUE`, `TENURE_YEARS_INT`, and `TOT_ERND_PREM_AMT`.
Every configured field is represented exactly once across the source and derived
dictionaries; disabled fields are included.

The build declares **24 derived outputs**. A fresh build omits the optional legacy
`ANCHOR_TENURE_YEARS_INT` output, yielding **416 columns**: 393 raw plus 23 derived.
Compatibility with an existing schema containing that legacy output can yield
**417 columns**. The numeric tenure output is passive compatibility metadata; the
categorical tenure group governs group selection.

`POLICY_TERM_END_DATE` is the scheduled term end. `EXPOSURE_END_DATE` is the actual
end of an exposure interval, which can reflect later cancellation or follow-up.
They serve different purposes and must not be substituted for one another.

## Reading the dictionaries

`original_config_dtype`, `var_type`, and `config_cv_ind` preserve the configuration
declarations. For build-only inputs and output-only fields absent from that
configuration, the original dtype and selection flag are blank. `snowflake_type`
describes the generalized warehouse contract.

The `role` column distinguishes required or identifying `structural` fields,
enabled `candidate` fields, `disabled` fields retained for completeness, and
`derived` outputs. A required structural field can also be an enabled feature;
`feature_candidate_enabled` records that separate fact. A selection flag does not
mean a field survived feature selection or belongs in the final model.

All identifiers are text, including keys whose legacy configuration used an
integer dtype. In particular, the exposure-row identifier can be alphanumeric.
Postal codes are text to preserve leading zeros. Dates use `DATE`. Continuous
measurements, amounts, and ratios use `FLOAT` where fractional arithmetic is
needed. Integer flags and counts use `NUMBER(38,0)` rather than narrow in-memory
integer limits. The dictionary explains any deliberate departure from the
original dtype. Numeric-looking category labels remain text when the configuration
declares them as strings.

The derived dictionary records the build's declared type families exactly.
`NUMBER` there is metadata about the output family; it is not an instruction to
round premiums, deductible conversions, or percentage changes to integers.

## Prediction-time availability

The prediction cutoff is the Month-1 evaluation date. **These dictionaries do not
verify the original warehouse's point-in-time correctness.** They distinguish
intended inputs from outcome data and identify fields needing upstream review.

`available_at_prediction=true` means the field is intended for use at Month 1
**only if its source value, source version, and every upstream calculation were
available by that date**. Its `availability_status` is conditional, and
`requires_upstream_timing_verification=true` makes that condition explicit. The
metadata is not evidence that this requirement was met in the historical data.

`available_at_prediction=false` means availability is unestablished or the field
is not approved for prediction. It also applies to disabled fields; it does not
assert that every disabled field is physically unknowable at Month 1. Read the
per-field `availability_note` before choosing inputs. A structural field being
available does not make it suitable as a model predictor.

Raw earned-premium, exposure, loss, and other measurement fields are marked false.
Selecting a row active at Month 1 does not establish when those values were
calculated. In particular, `WRITTEN_PREM` is a proxy derived from earned premium
divided by earned exposure. That proxy, its prior-term lag, and both rate-change
fields are also marked false until their source timing is established or they are
replaced with premium values known at Month 1. Their existing configuration
selection flags are preserved; availability metadata does not silently change
the pipeline's feature selection.

Month-13 retention and activity flags are future outcomes. Actual exposure end,
maximum exposure end, and retrospective history diagnostics support cohort or
label reconstruction and are excluded from prediction. The Month-13 evaluation
date is reserved for maturity checks and evaluation even when its scheduled date
could be calculated earlier.

## Creating and maintaining the contract

Provision your own authorized database and schema before executing the source
DDL. The DDL uses `CREATE TABLE IF NOT EXISTS`; it does not replace a table or
migrate an existing schema. All columns are nullable so a sparse synthetic SQL
fixture can populate only the required demonstration fields. Null placeholders
for unused predictors do not establish that a full modeling run has been
reproduced or validated.

The SQL is a Snowflake declaration. No live warehouse execution is implied by
the schema checks. Connection settings and load/render instructions are described
by the repository's data-acquisition documentation and examples.

Regenerate after intentional configuration or declared-contract changes:

```bash
python docs/schema/generate_schema.py
```

The generator rejects duplicate config fields, checks all configured and required
columns are covered, and prevents build-derived fields from appearing in the raw
source. It requires an explicit review when the declared derived outputs change.
It reads metadata only and does not import modeling modules or access Snowflake.
