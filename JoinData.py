import pandas as pd

Income = pd.read_csv('Data/Inkomen_van_huishoudens__regio_11122025_131729.csv', sep=';')
CoreFigure = pd.read_csv('Data/Regionale_kerncijfers_Nederland_11122025_133649.csv', sep=';')
PartyDistribution = pd.read_csv('Data/PartyDistribution.csv', sep=';')
ElectionResults = pd.read_csv('Data/uitslag_TK20251029_Gemeente.csv', sep=';')

# filter necessary columns

joined = pd.merge(
    Income, 
    CoreFigure, 
    how='inner', 
    left_on="Regio's", 
    right_on='Wijken en buurten'
)

joined.to_csv('Data/JoinedData1.csv', index=False)
