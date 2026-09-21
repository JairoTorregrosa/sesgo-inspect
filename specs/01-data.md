# 01 Data

Modules: `src/sesgo/_types.py` (exists, contract), `src/sesgo/_data.py` (conversion), `src/sesgo/_dataset.py` (samples). Console script: `sesgo-data`.

## Contract

### Input

Upstream repository `https://github.com/mvrobles/SESGO.git`, cloned to `external/SESGO` (git-ignored). Files are read from git objects with `git show <commit>:<path>`. The work tree is never checked out to an old commit.

| dataset_version | commit | language | files (`prompts/`) |
|---|---|---|---|
| `paper` | `89b8a0ef69fd86d7f08b99e06f43d16f1c8f599f` | `es` | `prompts_racismo.xlsx`, `prompts_genero.xlsx` (sheet `prompts_genero_es`), `prompts_clasismo.xlsx`, `prompts_xenofobia.xlsx` |
| `head` | `28b0a971152747a3b41222be36c446f95b9af9dc` | `es` | `prompts_{racismo,genero,clasismo,xenofobia}_es.xlsx` |
| `head` | same | `en` | `prompts_{racismo,clasismo,xenofobia}_en.xlsx`, `prompts_genero_EN.xlsx` |

`paper` + `en` does not exist upstream in complete form. The loader raises `ValueError` for it.

Verified facts for `paper`/`es` (2026-09-20, read-only pandas inspection):

| category | rows in file | filter | rows kept | ambig / disambig |
|---|---|---|---|---|
| racismo | 1318 | none | 1318 | 402 / 916 |
| genero | 684 | none | 684 | 228 / 456 |
| clasismo | 810 | none | 810 | 270 / 540 |
| xenofobia | 1524 | `bbq == False` | 1344 | UNVERIFIED after filter |

The xenofobia filter is `[UPSTREAM]`-derived: paper Table 1 lists 1344 original and 0 BBQ-adapted xenophobia prompts, and the published per-row results flag all 1344 rows as non-BBQ. The converter must assert each kept count. A mismatch is a hard error with the exact numbers.

Expected `head`/`es` counts: racismo 1086, genero 600, clasismo 810, xenofobia 1344 (no filter).

Columns used: `question_polarity` (`neg`/`nonneg`), `context_condition` (`ambig`/`disambig`), `answer_info` (Python-dict text with `ans0`,`ans1`,`ans2`; parse with `ast.literal_eval`), `context`, `question`, `label` (0/1/2 index of the correct option), `target` (always 1), `other` (always 0), `bbq` (bool). The upstream `category` column is ignored (it is wrong in some files); category comes from the file.

Role of each upstream option: index == `target` → `target`; index == `other` → `other`; remaining index → `unknown`. Assert that the `unknown` option text is a known unknown string ("No hay suficiente información" for `es`; for `en` derive and assert a single constant per file). Assert `label` role is `unknown` for every `ambig` row and `target` or `other` for every `disambig` row.

### Output files (git-ignored, under `data/`)

- `data/sesgo-{dataset_version}-{language}.jsonl`: one `Record` per line (see `_types.py`), UTF-8, `ensure_ascii=False`, sorted by id.
- `data/manifest.json`: for each file: upstream commit, per-category counts, per-split counts, SHA256 of the JSONL, converter version.

Record id: `{category}-{language}-{n:04d}`, `n` is the 1-based row position in the kept rows of that category file, in upstream row order. For `head` ids use the same form; the dataset version is not part of the id because a run uses one version. Ids are stable because the commit is pinned.

`options` are stored in upstream order (`ans0`, `ans1`, `ans2`) with their roles. Whitespace is stripped at both ends of `context`, `question`, and option texts. No other text change.

### Loader API (`_dataset.py`)

```python
def load_records(dataset_version: DatasetVersion = "paper", language: Language = "es",
                 data_dir: Path | None = None) -> list[Record]
def record_to_sample(record: Record, shuffle: bool) -> Sample
def sesgo_dataset(categories: Sequence[Category] | None, language: Language,
                  dataset_version: DatasetVersion, shuffle: bool,
                  limit_per_category: int | None, data_dir: Path | None = None) -> MemoryDataset
```

- `load_records` verifies the SHA256 against the manifest. If the file is absent it raises `FileNotFoundError` with the text `run: uv run sesgo-data build`. It does not clone or convert implicitly.
- Default `data_dir` is `<repo root>/data`, overridable by env `SESGO_DATA_DIR`.
- `sesgo-data build [--version paper|head|all] [--language es|en|all]` clones upstream if absent (`git clone`, then `git fetch`), verifies that both pinned commits exist, converts, writes manifest, prints counts. `sesgo-data verify` re-checks hashes and counts.

### Sample

- `id` = record id.
- `input` = `f"{context}\n\n{question}"` (plain; the solver builds the real prompt from metadata).
- `choices` = the three option texts in presentation order.
- `target` = letter (`A`/`B`/`C`) of the correct option in presentation order.
- `metadata` = `SampleMeta` fields (see `_types.py`): `category`, `language`, `dataset_version`, `context_condition`, `question_polarity`, `context`, `question`, `roles` (list of 3 roles aligned with `choices`), `label_role`, `bbq`, `category_split` (`f"{category}/{context_condition}"`).

### Shuffle (D4)

When `shuffle=True`: permutation = `random.Random(int.from_bytes(sha256(id.encode()).digest()[:8], "big")).sample(range(3), 3)` applied to options. `choices`, `roles`, and `target` all follow the same permutation. When `shuffle=False`: upstream order (other, target, unknown). Do not use `Dataset.shuffle_choices` (it does not remap metadata).

### limit_per_category

Deterministic stratified subset for pilots: inside each category, keep records in a balanced round-robin over the 4 cells (split × polarity) ordered by `sha256(id)`, until `limit_per_category` records. `None` keeps all. Values < 1 raise `ValueError`.

## Edge cases

- `answer_info` that fails to parse or lacks a key: hard error with file and row. (Upstream drops such rows; none exist at the pinned commits.)
- Duplicate ids or duplicate `(context, question, options)` inside a category: report the count in the manifest; do not drop.
- Unknown category or language: `ValueError` listing valid values.

## Acceptance

1. `uv run sesgo-data build --version all --language all` finishes and prints counts 1318/684/810/1344 (paper es) and 1086/600/810/1344 (head es).
2. `uv run sesgo-data verify` passes; editing one byte of a JSONL makes `load_records` fail with a hash error.
3. For every record: exactly one option per role; `label_role` rule holds.
4. With `shuffle=True`, the position of `unknown` over the full paper set is within 30–37 % for each of A, B, C, and two loads give identical samples.
5. For every sample, `roles[ord(target) - 65] == label_role`.
6. `git status` shows no file from `external/` or `data/`.
7. Unit tests cover: role derivation, shuffle determinism and remapping, `limit_per_category` balance, hash failure.
