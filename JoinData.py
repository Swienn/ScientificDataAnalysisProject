#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a municipality-level dataset for TK 2025 voting behavior analysis.

This version:
- Uses only necessary CSV files:
    * Income per region (CBS)
    * Regional core figures (CBS)
    * Wijk/buurt core figures (CBS) – ONLY for migration counts
    * Election results TK2025 per gemeente
    * PartyDistribution (to get parties with seats + economic & migration blocs)
- Computes per gemeente:
    * population (from core figures)
    * num_households
    * avg_income_household (euros)
    * median_income_household (euros)
    * pop_migration_background (count, from wijk/buurt)
    * pct_migration_background (percentage, 0–100)
    * per-party vote shares in percent, columns: <PARTY>_share_pct
    * bloc vote shares in percent:
        - econ_<ECON_CATEGORY>_share_pct
        - mig_<MIG_CATEGORY>_share_pct
    * votes_total (total valid votes used as denominator for shares)

Final output:
    Data/merged_tk2025_dataset.csv

Final dataset: 1 row per gemeente.
"""

import os
import re
import csv
import numpy as np
import pandas as pd


# helpers

def standardize_gemeente_name(series: pd.Series) -> pd.Series:
    """
    Clean municipality names:
    - strip whitespace & quotes
    - remove leading CBS codes like GMxxxx, BUxxxx, etc.
    - collapse spaces
    - uppercase
    """
    s = series.astype(str)
    s = s.str.strip().str.strip('"').str.strip("'")
    s = s.str.replace(r'^(GM|PV|BU|WK)\d+\s*', '', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s.str.upper().str.strip()


def add_gemeente_column(df: pd.DataFrame, candidates) -> pd.DataFrame:
    """
    Create 'gemeente_naam' from the first existing column in candidates.
    """
    for col in candidates:
        if col in df.columns:
            df["gemeente_naam"] = standardize_gemeente_name(df[col])
            return df
    raise ValueError(f"Could not find gemeente column. Columns: {list(df.columns)}")


def safe_read_cbs_csv(path: str) -> pd.DataFrame:
    """
    Robust read for CBS-style CSVs with weird quoting.
    """
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        thousands=".",
        engine="python",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )


def pick_first_matching_column(columns, patterns):
    """
    Find the first column whose name contains all substrings of any pattern.
    """
    cols_lower = {c: c.lower() for c in columns}
    for pattern in patterns:
        subs = [p.strip().lower() for p in pattern.split("/") if p.strip()]
        for col, col_l in cols_lower.items():
            if all(sub in col_l for sub in subs):
                return col
    return None


def sanitize_name(s: str) -> str:
    """
    Make a string safe for use in column names.
    Used for party names and category names.
    """
    s = str(s).upper()
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[\/\s]+", "_", s)
    s = re.sub(r"[^A-Z0-9_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


# Dataset loaders

def load_party_meta(path: str) -> pd.DataFrame:
    """
    Load PartyDistribution.csv and return a dataframe with:
        party, economic, migration

    The file has columns like:
        Party, Seats, Economic, Migration
    """
    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )
    cols = list(df.columns)

    party_col = pick_first_matching_column(cols, ["Party", "Partij", "PARTIJ"])
    econ_col = pick_first_matching_column(cols, ["Economic", "Economisch"])
    mig_col = pick_first_matching_column(cols, ["Migration", "Migratie"])

    if party_col is None:
        raise ValueError(f"Could not find a party column in PartyDistribution.csv. Columns: {cols}")

    # economic/migration may be None in theory, but in your file they exist
    party_meta = pd.DataFrame()
    party_meta["party"] = df[party_col].astype(str)
    if econ_col is not None:
        party_meta["economic"] = df[econ_col].astype(str)
    else:
        party_meta["economic"] = np.nan
    if mig_col is not None:
        party_meta["migration"] = df[mig_col].astype(str)
    else:
        party_meta["migration"] = np.nan

    return party_meta


def load_income(path: str) -> pd.DataFrame:
    """
    Load CBS income per region.

    Outputs:
    - gemeente_naam
    - num_households
    - avg_income_household      (mean disposable income, euros)
    - median_income_household   (median disposable income, euros)
    """
    df = safe_read_cbs_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    df = add_gemeente_column(df, ["Regio's", "Regio", "Gemeentenaam", "Gemeente"])

    cols = list(df.columns)
    col_hh = pick_first_matching_column(cols, ["Particuliere huishoudens (x 1 000)"])
    col_avg = pick_first_matching_column(cols, ["Inkomen/Gemiddeld besteedbaar inkomen (1 000 euro)"])
    col_med = pick_first_matching_column(cols, ["Inkomen/Mediaan besteedbaar inkomen (1 000 euro)"])

    if col_hh is None or col_avg is None or col_med is None:
        raise ValueError(
            f"Missing income columns. Found columns: {cols}"
        )

    df["num_households"] = pd.to_numeric(df[col_hh], errors="coerce") * 1000
    df["avg_income_household"] = pd.to_numeric(df[col_avg], errors="coerce") * 1000
    df["median_income_household"] = pd.to_numeric(df[col_med], errors="coerce") * 1000

    return df[["gemeente_naam", "num_households", "avg_income_household", "median_income_household"]]


def load_core_population(path: str) -> pd.DataFrame:
    """
    Load CBS regional core figures and keep total population per gemeente.
    """
    df = safe_read_cbs_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    df = add_gemeente_column(df, ["Regio's", "Regio", "Gemeentenaam", "Gemeente"])

    col_pop = pick_first_matching_column(
        df.columns,
        ["Totale bevolking (aantal)"]
    )
    if col_pop is None:
        raise ValueError("Could not find population column in core figures CSV.")

    df["pop_total_core"] = pd.to_numeric(df[col_pop], errors="coerce")

    df = df[df["pop_total_core"].notna()]

    return df[["gemeente_naam", "pop_total_core"]]


def load_wijk_buurt(path: str) -> pd.DataFrame:
    """
    Load wijk/buurt file and aggregate to gemeente level for migration counts.

    We explicitly remove:
    - the NL total row
    - the gemeente-total rows

    We do NOT use this file for total population in the final dataset;
    population comes from core figures instead.
    """
    df = safe_read_cbs_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    col_regdesc = pick_first_matching_column(df.columns, ["Wijken en buurten"])
    col_gem = pick_first_matching_column(df.columns, ["Regioaanduiding/Gemeentenaam (naam)"])
    col_eu = pick_first_matching_column(df.columns, ["Herkomstland/Europa (exclusief Nederland)"])
    col_non_eu = pick_first_matching_column(df.columns, ["Herkomstland/Buiten Europa"])

    if any(x is None for x in [col_regdesc, col_gem, col_eu, col_non_eu]):
        raise ValueError("Could not find all required wijk/buurt columns.")

    df = add_gemeente_column(df, [col_gem])

    # Clean region & gemeente strings properly
    region_clean = df[col_regdesc].astype(str).str.strip('"').str.strip()
    gem_clean = df[col_gem].astype(str).str.strip('"').str.strip()

    mask_national = (region_clean == "Nederland") & (gem_clean == "Nederland")
    mask_gemeente_total = region_clean == gem_clean

    df_detail = df[~(mask_national | mask_gemeente_total)].copy()

    df_detail["pop_migration_background"] = (
        pd.to_numeric(df_detail[col_eu], errors="coerce") +
        pd.to_numeric(df_detail[col_non_eu], errors="coerce")
    )

    grouped = df_detail.groupby("gemeente_naam", as_index=False).sum(numeric_only=True)

    return grouped[["gemeente_naam", "pop_migration_background"]]


def load_election_results(path: str, party_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Load election results and:
    - keep only parties that have seats (from PartyDistribution)
    - aggregate to gemeente level
    - compute:
        * per-party vote shares in percent, columns: <PARTY>_share_pct
        * per economic bloc vote shares in percent: econ_<CAT>_share_pct
        * per migration bloc vote shares in percent: mig_<CAT>_share_pct
        * votes_total
    """
    valid_parties = party_meta["party"].dropna().unique().tolist()

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )

    df = add_gemeente_column(df, ["Gemeente"])

    df = df[df["Partij"].isin(valid_parties)]

    df["AantalStemmen"] = pd.to_numeric(df["AantalStemmen"], errors="coerce").fillna(0)

    # Pivot to votes per party
    pivot_votes = df.pivot_table(
        index="gemeente_naam",
        columns="Partij",
        values="AantalStemmen",
        aggfunc="sum",
        fill_value=0,
    )

    # Total votes per gemeente
    votes_total = pivot_votes.sum(axis=1)

    # Shares in PERCENT (0–100)
    share_df = pivot_votes.div(votes_total.replace(0, np.nan), axis=0) * 100

    # Result dataframe we'll return
    result = pd.DataFrame(index=share_df.index)

    # Per-party share columns
    for party in share_df.columns:
        col_name = sanitize_name(party) + "_share_pct"
        result[col_name] = share_df[party]

    # Economic bloc shares
    if "economic" in party_meta.columns:
        econ_cats = party_meta["economic"].dropna().unique()
        for cat in econ_cats:
            parties_cat = party_meta.loc[party_meta["economic"] == cat, "party"]
            parties_cat = [p for p in parties_cat if p in share_df.columns]
            if not parties_cat:
                continue
            col_name = "econ_" + sanitize_name(cat) + "_share_pct"
            result[col_name] = share_df[parties_cat].sum(axis=1)

    # Migration bloc shares
    if "migration" in party_meta.columns:
        mig_cats = party_meta["migration"].dropna().unique()
        for cat in mig_cats:
            parties_cat = party_meta.loc[party_meta["migration"] == cat, "party"]
            parties_cat = [p for p in parties_cat if p in share_df.columns]
            if not parties_cat:
                continue
            col_name = "mig_" + sanitize_name(cat) + "_share_pct"
            result[col_name] = share_df[parties_cat].sum(axis=1)

    # Add votes_total (absolute)
    result["votes_total"] = votes_total

    return result.reset_index()


def main():
    data_dir = "Data"

    income_file = os.path.join(data_dir, "Inkomen_van_huishoudens__regio_11122025_135329.csv")
    core_file = os.path.join(data_dir, "Regionale_kerncijfers_Nederland_11122025_133649.csv")
    wb_file = os.path.join(data_dir, "Kerncijfers_wijken_en_buurten_2024_11122025_141648.csv")
    party_file = os.path.join(data_dir, "PartyDistribution.csv")
    election_file = os.path.join(data_dir, "uitslag_TK20251029_Gemeente.csv")

    print("Loading party metadata (parties with seats + blocs)...")
    party_meta = load_party_meta(party_file)
    print("Parties with seats:", party_meta["party"].unique())

    print("Loading income data...")
    income = load_income(income_file)
    print("Income shape:", income.shape)

    print("Loading core population...")
    core = load_core_population(core_file)
    print("Core population shape:", core.shape)

    print("Loading wijk/buurt migration aggregation...")
    wb = load_wijk_buurt(wb_file)
    print("Wijk/buurt aggregated shape:", wb.shape)

    print("Loading election results and computing shares...")
    election = load_election_results(election_file, party_meta)
    print("Election shape:", election.shape)

    print("Merging datasets...")
    df = income.merge(core, on="gemeente_naam", how="inner")
    df = df.merge(wb, on="gemeente_naam", how="inner")
    df = df.merge(election, on="gemeente_naam", how="inner")

    # Rename population column and compute migration percentage
    df = df.rename(columns={"pop_total_core": "population"})
    df["pct_migration_background"] = (
        100 * df["pop_migration_background"] / df["population"].replace(0, np.nan)
    )

    # Put key variables up front
    first_cols = [
        "gemeente_naam",
        "population",
        "num_households",
        "avg_income_household",
        "median_income_household",
        "pop_migration_background",
        "pct_migration_background",
        "votes_total",
    ]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    assert df["gemeente_naam"].is_unique, "Duplicate gemeente rows in final dataset!"

    output = os.path.join(data_dir, "merged_tk2025_dataset.csv")
    df.to_csv(output, index=False)

    print("\nDone! Saved:", output)
    print("Final shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
