import pandas as pd
import os

# Filstier
md_dir       = os.path.join('metadata')
rf_path      = os.path.join(md_dir, 'metadata_rf.csv')
audio_path   = os.path.join(md_dir, 'metadata_audio.csv')
out_path     = os.path.join(md_dir, 'metadata.csv')

# Les inn metadata
df_rf    = pd.read_csv(rf_path)
df_audio = pd.read_csv(audio_path)

# Sjekk hvilke kolonner vi har
print("RF-kolonner:", df_rf.columns.tolist())
print("Audio-kolonner:", df_audio.columns.tolist())

# Slå sammen på segment-ID (endre 'segment_id' om nødvendig)
df_meta = pd.merge(df_rf,
                   df_audio,
                   on='segment_id',
                   how='inner',   # -> velg 'outer' for å beholde alle, så kan du filtrere
                   suffixes=('_rf', '_audio'))

# Eventuelt: rapporter manglende par
n_rf    = df_rf['segment_id'].nunique()
n_audio = df_audio['segment_id'].nunique()
n_meta  = df_meta['segment_id'].nunique()
print(f"Unike segment i RF:    {n_rf}")
print(f"Unike segment i Audio: {n_audio}")
print(f"Matched segment:       {n_meta}")

# Skriv ut
df_meta.to_csv(out_path, index=False)
print(f"metadata.csv skrevet til {out_path}")
