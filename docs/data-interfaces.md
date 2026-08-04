# Scalable Data Interface Contract

This document is authoritative for bulk prediction input, strict CSV import,
preprocessing, dataset splitting, and missing values.

## Bulk Prediction Input

`predict PROJECT --input FILE|- [--batch-size N]` consumes this version 1
document:

```text
neural-c inputs 1
samples 3
inputs 4
sample 0 5.1 3.5 1.4 0.2
sample 1 6.4 3.2 4.5 1.5
sample 2 6.3 3.3 6.0 2.5
end
```

Counts are positive, sample indices are contiguous from zero, each row has the
declared width, and no content may follow `end`. `-` selects standard input.
The execution-only batch size defaults to 1024 and bounds input/output working
memory. Output remains ordered and byte-identical to positional prediction and
across worker or batch counts. The command stages output until the complete
input has validated, so a late malformed row never exposes a partial result.

## Explicit CSV Schema

Import uses `import-csv PROJECT CSV --schema FILE`. It takes the exclusive
project lock and refuses to invalidate an existing `weights.txt` or
`checkpoint.txt`. The schema never guesses a header, input, target, or label:

```text
neural-c csv-schema 1
columns 5
header yes
inputs 0 1 2 3
label 4
class setosa 1 0 0
class versicolor 0 1 0
class virginica 0 0 1
end
```

Numeric targets instead use `targets COLUMN...` and must match the model output
width. Column indices are zero-based. Categorical names are single exact UTF-8
tokens and every class line supplies the complete numeric output vector.

CSV fields support unquoted text, quoted fields, and doubled quotes. Embedded
newlines are deliberately unsupported. Every row must have exactly `columns`
fields. Numbers use the C decimal grammar regardless of process locale.
Unknown labels, invalid targets, malformed quotes, and row-width mismatches
identify the source row and fail before managed data is replaced.

## Split Ownership and Provenance

`--validation-ratio R`, `--test-ratio R`, and `--split-seed N` control a
reproducible split; both ratios default to zero and their sum must be below one.
Categorical imports shuffle each declared class independently with neural-c's
specified SplitMix64 stream, which makes the split stratified. Numeric-target
imports shuffle the complete dataset. Subset sizes use `floor(class_size *
ratio)` for each categorical class or `floor(total_size * ratio)` otherwise.
Requested nonzero subsets that cannot receive a sample fail actionably.

Rows within each generated native dataset retain source order after assignment.
The CSV and schema SHA-256 digests, requested ratios, seed, and stratification
flag are persisted in `preprocessing.txt`. `train.txt`, optional
`validation.txt`, optional `test.txt`, and `preprocessing.txt` are staged before
replacement and installed as one rollback-capable project transaction.

## Normalization and Missing Values

`--normalization none|standardize|minmax` defaults to `none`.
`standardize` persists training mean and population standard deviation;
`minmax` persists training minimum and range. Constant features use scale 1.
Statistics are fitted exclusively from the training subset, then applied to
training, validation, and test rows before native datasets are stored.

`--missing reject|mean` defaults to `reject`. CSV input fields use an empty
field or exact `?` as a missing value. Targets can never be missing. `reject`
fails if any generated subset contains a missing input. `mean` computes each
imputation value from finite training values only and fails if a training
feature has none. Imputation precedes normalization.

`preprocessing.txt` is a versioned project-owned file. Prediction loads it into
the same immutable snapshot as the model and weights and applies the exact
persisted transform. Positional or bulk prediction can use exact `?` only when
the persisted policy permits it. Projects without the file retain the legacy
identity behavior and reject all non-finite or missing inputs. Evaluation does
not transform again because imported native datasets are already transformed.

Preprocessing metadata participates in dataset provenance whenever present,
so changing it invalidates weights and checkpoints. Its absence preserves the
legacy canonical dataset digest exactly.
