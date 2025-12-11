import pandas as pd

df = pd.read_csv(
    "Data/Kerncijfers_wijken_en_buurten_2024_11122025_141648.csv",
    sep=";",
    decimal=",",
    thousands=".",
    engine="python"
)

out = df.groupby("Regioaanduiding/Gemeentenaam (naam)", as_index=False).sum(numeric_only=True)
out.to_csv("Data/migration_by_regio_gemeente.csv", index=False)
