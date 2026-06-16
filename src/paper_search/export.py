from __future__ import annotations

from io import BytesIO

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "papers") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        safe_sheet = sheet_name[:31] or "papers"
        df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return buffer.getvalue()
