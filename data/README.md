# Dataset access and attribution

**UCI Bank Marketing** — dataset ID 222.

- Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- User-provided archive URL: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
- Source license: **CC BY 4.0**; attribute the authors and original data provider when using or redistributing.

The notebook uses `bank-full.csv` (semicolon-separated, **45,211** rows, **16** predictors + `y` target). Do not confuse it with `bank-additional-full.csv`, a different **41,188**-row dataset variant.

## Download

From the repository root:

```shell
python scripts/download_data.py
```

The script verifies archive contents, selects `bank-full.csv` (handling an outer UCI archive that contains a nested `bank.zip`), and places it in `notebooks/bank-full.csv` because the historical notebook uses `pd.read_csv('bank-full.csv', sep=';')`.

This directory and file are excluded from version control. If network access is unavailable, manually extract `bank-full.csv` from the official UCI `bank.zip` and copy it to `notebooks/bank-full.csv`.

`duration` measures the last contact duration and is **not a pre-call feature**. UCI specifically cautions about using it in realistic pre-contact prediction. `pdays = -1` indicates no previous contact.
