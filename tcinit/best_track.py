"""BestTrack — normalized access to JTWC (ATCF) and NHC (HURDAT2) best-track archives.

The class ingests either format and normalizes to a common HURDAT2-based
schema. Downstream code should use the per-datetime SI accessors
(:meth:`get_location`, :meth:`get_windspeed`, :meth:`get_central_pressure`,
:meth:`get_30kt_radius`) rather than reaching into ``df_best_tracks_history``.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Optional
import csv
import os
import re

import numpy as np
import pandas as pd

from tcinit.constants import HPA_TO_PA, KNOTS_TO_MPS, NM_TO_KM

# MSLP plausibility bounds (hPa). Values outside this range are treated as
# sentinel garbage: ATCF files carry -999 as "missing", but the corrupt-row
# survey found 0 and 9-digit integers in the wild. 850 hPa is below the
# strongest observed TCs (Tip 1979: 870 hPa); 1100 hPa is well above any
# realistic surface pressure.
_MSLP_MIN_HPA = 850.0
_MSLP_MAX_HPA = 1100.0


class BestTrack:
    """Normalized best-track data from JTWC (ATCF) or NHC (HURDAT2) sources.

    Both formats are normalized to a HURDAT2-style DataFrame keyed by an
    integer time step. The core columns available on every row are
    ``datetimes``, ``LAT_DEG``, ``LON_DEG`` (signed float degrees; longitude
    wrapped to [0, 360)), ``VMAX`` (knots), ``MSLP`` (hPa; -999 sentinel),
    and the ``RAD34_*`` wind-radii quadrants (nm; blank for ATCF sources).

    See the source of the prototype (causaltc/data_collection/best_track.py)
    for the full column dictionary if you need to reach into the DataFrame
    directly.
    """

    STATUS_RANK: dict[str, int] = {
        # Low / non-tropical
        "DB": 0,
        "WV": 0,
        "LO": 0,
        "EX": 0,
        # Sub-tropical / depression
        "SD": 1,
        "TD": 1,
        # Gale / storm
        "SS": 2,
        "TS": 2,
        "ST": 2,  # ST = severe tropical storm (JMA)
        # Hurricane / typhoon
        "HU": 3,
        "TY": 3,
        "CY": 3,
        "TC": 3,
        # Super-typhoon
        "STY": 4,
        "SU": 4,
    }

    def __init__(self, best_tracks_csv: str):
        """Load and normalize a best-track file.

        Args:
            best_tracks_csv: Path to a best-track file (HURDAT2 ``.txt`` or
                ATCF ``.dat``). Filename must match ``b<basin><NN><YYYY>.*``
                so the basin/number/year metadata can be parsed.
        """
        self.best_tracks_csv = best_tracks_csv
        self.format_type = self.detect_format(best_tracks_csv)
        self.basin_long, self.basin_short, self.number, self.ID, self.year = (
            self.extract_variables(best_tracks_csv)
        )
        df_raw = self.load_best_tracks_csv(best_tracks_csv)
        self.df_best_tracks_history = self.normalize_to_hurdat2_format(df_raw)
        self._add_datetimes_column()

    @classmethod
    def from_root_and_id(
        cls,
        best_track_root: Union[str, Path],
        cyclone_id: str,
    ) -> "BestTrack":
        """Find ``{cyclone_id}.*`` recursively under ``best_track_root``."""
        root = Path(best_track_root)
        try:
            bt_file = next(root.rglob(f"{cyclone_id}.*"))
        except StopIteration as exc:
            raise FileNotFoundError(
                f"No best-track file for cyclone_id='{cyclone_id}' under {root}"
            ) from exc
        return cls(str(bt_file))

    def detect_format(self, file_path: str) -> str:
        """Return ``'atcf'`` or ``'hurdat2'`` based on the first non-empty row."""
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                # HURDAT2: first col is 8-digit YYYYMMDD, second is <=4-digit HHMM.
                if len(parts[0]) == 8 and parts[0].isdigit():
                    if len(parts) > 1 and len(parts[1]) <= 4 and parts[1].isdigit():
                        return "hurdat2"
                # ATCF: third col is 10-digit YYYYMMDDHH.
                if len(parts) > 2 and len(parts[2]) == 10 and parts[2].isdigit():
                    return "atcf"
                raise ValueError(
                    f"Unable to detect format from file: {file_path}\n"
                    f"First line: {line}"
                )
        raise ValueError(f"Empty best-track file: {file_path}")

    def _latest_dt_at_extreme(
        self,
        series: pd.Series,
        extrema: str,
        df: pd.DataFrame,
    ) -> Optional[pd.Timestamp]:
        """Latest datetime where ``series`` attains its extreme (``'max'``/``'min'``)."""
        if series.empty or series.isna().all():
            return None
        if extrema == "max":
            extreme_val = series.max()
        elif extrema == "min":
            extreme_val = series.min()
        else:
            raise ValueError("extrema must be 'max' or 'min'")
        dts = df.loc[series == extreme_val, "datetimes"]
        return dts.max() if not dts.empty else None

    def get_peak_intensity_datetime(self) -> Optional[datetime]:
        """Latest datetime at which the cyclone reaches any peak criterion.

        Considers three: max VMAX, min MSLP, max STATUS rank. Ties within a
        criterion are resolved by the latest datetime; the return is the
        latest of the three per-criterion peaks.
        """
        df = self.df_best_tracks_history
        if "datetimes" not in df.columns:
            raise KeyError("Column 'datetimes' missing; BestTrack not initialised.")

        peak_datetimes: List[pd.Timestamp] = []
        vmax = pd.to_numeric(df["VMAX"], errors="coerce")
        dt_vmax = self._latest_dt_at_extreme(vmax, "max", df)
        if dt_vmax is not None:
            peak_datetimes.append(dt_vmax)

        mslp = pd.to_numeric(df["MSLP"].replace(-999, np.nan), errors="coerce")
        dt_mslp = self._latest_dt_at_extreme(mslp, "min", df)
        if dt_mslp is not None:
            peak_datetimes.append(dt_mslp)

        status = df["STATUS"].astype(str).str.strip().str.upper()
        rank = status.map(self.STATUS_RANK).astype(float)
        dt_status = self._latest_dt_at_extreme(rank, "max", df)
        if dt_status is not None and not np.isnan(rank.max()):
            peak_datetimes.append(dt_status)

        return max(peak_datetimes) if peak_datetimes else None

    def extract_variables(self, file_path: str):
        """Extract ``(basin_long, basin_short, number, ID, year)`` from the path."""
        base_name = os.path.basename(file_path)
        match = re.match(r"(\w+)(\d{2})(\d{4})\.*", base_name)
        if not match:
            raise ValueError(f"file_path format is incorrect: {file_path}")
        path_parts = file_path.split("/")
        # Basin-long from directory structure is best-effort; safe default:
        basin_long = path_parts[4] if len(path_parts) > 4 else "unknown"
        basin_short = match.group(1)[1:3]
        number = match.group(2)
        year = match.group(3)
        ID = f"b{basin_short}{number}{year}"
        return basin_long, basin_short, number, ID, year

    def load_best_tracks_csv(self, file_path: str) -> pd.DataFrame:
        """Load rows into a DataFrame with format-appropriate column names."""
        all_rows = []
        with open(file_path, "r") as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    all_rows.append(row)
        if not all_rows:
            raise ValueError(f"No valid data rows found in file: {file_path}")

        max_columns = max(len(row) for row in all_rows)
        padded_rows = [row + [""] * (max_columns - len(row)) for row in all_rows]

        if self.format_type == "atcf":
            doc_names = [
                "BASIN",
                "CY",
                "YYYYMMDDHH",
                "TECHNUM",
                "TECH",
                "TAU",
                "LAT",
                "LON",
                "VMAX",
                "MSLP",
                "TY",
                "RAD",
                "WINDCODE",
                "RAD1",
                "RAD2",
                "RAD3",
                "RAD4",
                "POUTER",
                "ROUTER",
                "RMW",
                "GUSTS",
                "EYE",
                "SUBREGION",
                "MAXSEAS",
                "INITIALS",
                "DIR",
                "SPEED",
                "STORMNAME",
                "DEPTH",
                "SEAS",
                "SEASCODE",
                "SEAS1",
                "SEAS2",
                "SEAS3",
                "SEAS4",
            ]
        else:  # hurdat2
            doc_names = [
                "YYYYMMDD",
                "HHMM",
                "RECORD",
                "STATUS",
                "LAT",
                "LON",
                "VMAX",
                "MSLP",
                "RAD34_NE",
                "RAD34_SE",
                "RAD34_SW",
                "RAD34_NW",
                "RAD50_NE",
                "RAD50_SE",
                "RAD50_SW",
                "RAD50_NW",
                "RAD64_NE",
                "RAD64_SE",
                "RAD64_SW",
                "RAD64_NW",
            ]

        if max_columns > len(doc_names):
            extra_names = [f"EXTRA_{i+1}" for i in range(max_columns - len(doc_names))]
            col_names = doc_names + extra_names
        else:
            col_names = doc_names[:max_columns]

        return pd.DataFrame(padded_rows, columns=col_names)

    def normalize_to_hurdat2_format(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Return a copy normalized to the HURDAT2 column schema."""
        if self.format_type == "hurdat2":
            return df_raw.copy()

        df_normalized = pd.DataFrame()
        num_rows = len(df_raw)

        df_normalized["YYYYMMDD"] = df_raw["YYYYMMDDHH"].astype(str).str.strip().str[:8]
        df_normalized["HHMM"] = (
            df_raw["YYYYMMDDHH"].astype(str).str.strip().str[8:10].str.zfill(2) + "00"
        )
        df_normalized["RECORD"] = [""] * num_rows
        df_normalized["STATUS"] = df_raw["TY"]
        df_normalized["LAT"] = df_raw["LAT"].apply(self._convert_tenths_to_latlon)
        df_normalized["LON"] = df_raw["LON"].apply(self._convert_tenths_to_latlon)
        df_normalized["VMAX"] = df_raw["VMAX"]
        df_normalized["MSLP"] = df_raw["MSLP"]

        for col in [
            "RAD34_NE",
            "RAD34_SE",
            "RAD34_SW",
            "RAD34_NW",
            "RAD50_NE",
            "RAD50_SE",
            "RAD50_SW",
            "RAD50_NW",
            "RAD64_NE",
            "RAD64_SE",
            "RAD64_SW",
            "RAD64_NW",
        ]:
            df_normalized[col] = [""] * num_rows

        for col in [
            "BASIN",
            "CY",
            "YYYYMMDDHH",
            "TECHNUM",
            "TECH",
            "TAU",
            "RAD",
            "WINDCODE",
            "RAD1",
            "RAD2",
            "RAD3",
            "RAD4",
            "POUTER",
            "ROUTER",
            "RMW",
            "GUSTS",
            "EYE",
            "SUBREGION",
            "MAXSEAS",
            "INITIALS",
            "DIR",
            "SPEED",
            "STORMNAME",
            "DEPTH",
            "SEAS",
            "SEASCODE",
            "SEAS1",
            "SEAS2",
            "SEAS3",
            "SEAS4",
        ]:
            if col in df_raw.columns:
                df_normalized[col] = df_raw[col]

        for col in df_raw.columns:
            if col.startswith("EXTRA_"):
                df_normalized[col] = df_raw[col]

        return df_normalized

    def _convert_tenths_to_latlon(self, coord_str: str) -> str:
        """Convert ATCF tenths format (``231N``) to HURDAT2 decimal (``23.1N``)."""
        coord_str = str(coord_str).strip()
        if not coord_str or coord_str in ["-999", ""]:
            return coord_str
        try:
            hemisphere = coord_str[-1]
            numeric_part = int(coord_str[:-1])
            decimal_value = numeric_part / 10.0
            return f"{decimal_value:.1f}{hemisphere}"
        except (ValueError, IndexError):
            return coord_str

    def _add_datetimes_column(self):
        """Attach a ``datetimes`` (pd.Timestamp) column and numeric LAT/LON_DEG."""
        df = self.df_best_tracks_history

        if self.format_type == "atcf":
            date_col = "YYYYMMDDHH"
            if date_col not in df.columns:
                raise KeyError(f"Expected column '{date_col}' for ATCF data.")
            dt_series = pd.to_datetime(
                df[date_col].astype(str).str.strip(), format="%Y%m%d%H"
            )
            insert_after = date_col
        else:  # hurdat2
            date_col, time_col = "YYYYMMDD", "HHMM"
            missing = {date_col, time_col} - set(df.columns)
            if missing:
                raise KeyError(
                    f"Missing HURDAT2 date/time columns: {', '.join(missing)}"
                )
            hhmm = df[time_col].astype(str).str.zfill(4).str.strip()
            yyyymmdd = df[date_col].astype(str).str.strip()
            dt_series = pd.to_datetime(yyyymmdd + hhmm, format="%Y%m%d%H%M")
            insert_after = date_col

        insert_idx = df.columns.get_loc(insert_after) + 1
        df.insert(insert_idx, "datetimes", dt_series)
        self.df_best_tracks_history = df
        self._add_numeric_coordinates()

    def _add_numeric_coordinates(self) -> None:
        """Append LAT_DEG / LON_DEG columns (signed floats; LON wrapped to [0, 360))."""

        def _parse_coord(val) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            if not s:
                return float("nan")
            hemi = s[-1].upper() if s[-1].isalpha() else ""
            try:
                number = float(s[:-1]) if hemi else float(s)
            except ValueError:
                return float("nan")
            if hemi in ("S", "W"):
                number = -number
            return number

        df = self.df_best_tracks_history
        if "LAT_DEG" not in df.columns:
            self.df_best_tracks_history["LAT_DEG"] = df["LAT"].apply(_parse_coord)
        if "LON_DEG" not in df.columns:
            lon = df["LON"].apply(_parse_coord)
            self.df_best_tracks_history["LON_DEG"] = lon.mod(360)

    def get_datetimes(self) -> List[datetime]:
        """Sorted list of datetimes covering the cyclone's history."""
        return sorted(self.df_best_tracks_history["datetimes"].tolist())

    def get_intensity_loc_data(self) -> pd.DataFrame:
        """DataFrame view of ``[datetimes, LAT, LON, STATUS, VMAX, MSLP]``."""
        df = self.df_best_tracks_history[
            ["datetimes", "LAT", "LON", "STATUS", "VMAX", "MSLP"]
        ]
        return df.copy()

    # ------------------------------------------------------------------
    # Per-datetime SI accessors — these are what the bogus vortex reads.
    # ------------------------------------------------------------------

    def _row_at_datetime(self, target_time: datetime) -> pd.Series:
        """First best-track row whose ``datetimes`` matches ``target_time``."""
        df = self.df_best_tracks_history
        matches = df.loc[df["datetimes"] == target_time]
        if matches.empty:
            raise ValueError(f"BestTrack {self.ID}: no row at datetime {target_time}")
        return matches.iloc[0]

    def get_location(self, target_time: datetime) -> Tuple[float, float]:
        """``(lat_deg, lon_deg)`` at ``target_time``. Longitude wrapped to [0, 360)."""
        row = self._row_at_datetime(target_time)
        lat_deg = float(row["LAT_DEG"])
        lon_deg = float(row["LON_DEG"])
        if not (np.isfinite(lat_deg) and np.isfinite(lon_deg)):
            raise ValueError(
                f"BestTrack {self.ID}: implausible location "
                f"(LAT_DEG={lat_deg}, LON_DEG={lon_deg}) at {target_time}"
            )
        return lat_deg, lon_deg

    def get_windspeed(self, target_time: datetime) -> float:
        """Maximum sustained wind (m/s) at ``target_time``."""
        row = self._row_at_datetime(target_time)
        vmax_kt = float(row["VMAX"])
        if not np.isfinite(vmax_kt) or vmax_kt <= 0:
            raise ValueError(
                f"BestTrack {self.ID}: implausible VMAX={vmax_kt!r} kt at {target_time}"
            )
        return vmax_kt * KNOTS_TO_MPS

    def get_central_pressure(self, target_time: datetime) -> float:
        """Minimum central pressure (Pa) at ``target_time``.

        Filters MSLP to [850, 1100] hPa; anything outside raises ``ValueError``.
        This screens out the ATCF sentinel values (``-999``, ``0``, 9-digit
        garbage) surfaced by the corrupt-row survey.
        """
        row = self._row_at_datetime(target_time)
        mslp_hpa = float(row["MSLP"])
        if not (_MSLP_MIN_HPA <= mslp_hpa < _MSLP_MAX_HPA):
            raise ValueError(
                f"BestTrack {self.ID}: implausible MSLP={mslp_hpa!r} hPa "
                f"at {target_time} (expected [{_MSLP_MIN_HPA}, {_MSLP_MAX_HPA}))"
            )
        return mslp_hpa * HPA_TO_PA

    def get_30kt_radius(self, target_time: datetime) -> Optional[float]:
        """Mean radius of 30-kt wind (km) across the four RAD34_* quadrants.

        Returns ``None`` when the columns are absent or every quadrant is
        blank / non-finite / non-positive (the typical ATCF-source state).
        """
        row = self._row_at_datetime(target_time)
        cols = ("RAD34_NE", "RAD34_SE", "RAD34_SW", "RAD34_NW")
        if not all(c in row.index for c in cols):
            return None
        quadrants_nm = []
        for c in cols:
            try:
                v = float(row[c])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                quadrants_nm.append(v)
        if not quadrants_nm:
            return None
        return float(np.mean(quadrants_nm)) * NM_TO_KM
