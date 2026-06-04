from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

RASTER = Path("outputs/classified_map.tif")

PIXEL_SIZE_M = 20
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M
PIXEL_AREA_HA = PIXEL_AREA_M2 / 10000

CLASS_NAMES = {
    0: "Agua",
    1: "Arena",
    2: "Playa_sal",
}

with rasterio.open(RASTER) as src:
    data = src.read(1)

unique, counts = np.unique(data, return_counts=True)

rows = []

for value, count in zip(unique, counts):

    if value == 255:
        continue

    area_m2 = count * PIXEL_AREA_M2
    area_ha = count * PIXEL_AREA_HA
    area_km2 = area_m2 / 1_000_000

    rows.append(
        {
            "class_id": int(value),
            "class_name": CLASS_NAMES.get(int(value), f"Clase_{value}"),
            "pixels": int(count),
            "area_m2": round(area_m2, 2),
            "area_ha": round(area_ha, 4),
            "area_km2": round(area_km2, 6),
        }
    )

df = pd.DataFrame(rows)

print()
print(df)
print()

csv_path = Path("outputs/class_areas.csv")
df.to_csv(csv_path, index=False)

print(f"CSV exportado: {csv_path}")