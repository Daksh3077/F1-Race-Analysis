"""
OpenF1 API data loader — replaces FastF1 for race analysis.
Covers 2023 onwards. Fast, lightweight, no caching needed.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st

BASE = "https://api.openf1.org/v1"


def _get(endpoint, **params):
    """Make a GET request to OpenF1 API."""
    try:
        resp = requests.get(f"{BASE}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"OpenF1 API error ({endpoint}): {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_seasons():
    """Returns list of available years (2023 onwards)."""
    return [2023, 2024, 2025, 2026]


@st.cache_data(show_spinner=False)
def get_races(year):
    """Get all race sessions for a given year."""
    df = _get("sessions", year=year, session_type="Race")
    if df.empty:
        return []
    return df[["session_key", "session_name", "location",
               "country_name", "date_start"]].to_dict("records")


@st.cache_data(show_spinner=False)
def get_session_key(year, location):
    """Get the session key for a specific race."""
    df = _get("sessions", year=year, session_type="Race",
              location=location)
    if df.empty:
        return None
    return int(df.iloc[0]["session_key"])


@st.cache_data(show_spinner=False)
def get_drivers(session_key):
    """Get all drivers for a session."""
    df = _get("drivers", session_key=session_key)
    if df.empty:
        return df
    cols = ["driver_number", "name_acronym", "full_name",
            "team_name", "team_colour"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates("driver_number")


@st.cache_data(show_spinner=False)
def get_laps(session_key):
    """Get all lap data for a session."""
    df = _get("laps", session_key=session_key)
    if df.empty:
        return df

    # Convert lap duration to seconds
    if "lap_duration" in df.columns:
        df["LapTimeSeconds"] = pd.to_numeric(
            df["lap_duration"], errors="coerce")

    # Rename columns to match existing chart code
    rename = {
        "driver_number":    "DriverNumber",
        "lap_number":       "LapNumber",
        "lap_duration":     "LapDuration",
        "duration_sector_1":"Sector1",
        "duration_sector_2":"Sector2",
        "duration_sector_3":"Sector3",
        "compound":         "Compound",
        "tyre_age_at_start":"TyreLife",
        "stint_number":     "Stint",
        "is_pit_out_lap":   "IsPitOutLap",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


@st.cache_data(show_spinner=False)
def get_positions(session_key):
    """Get position data (lap-by-lap position per driver)."""
    df = _get("position", session_key=session_key)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_pit_stops(session_key):
    """Get pit stop data."""
    return _get("pit", session_key=session_key)


@st.cache_data(show_spinner=False)
def get_weather(session_key):
    """Get weather data."""
    df = _get("weather", session_key=session_key)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_telemetry(session_key, driver_number):
    """Get car telemetry for a specific driver (speed, throttle, etc)."""
    df = _get("car_data", session_key=session_key,
              driver_number=driver_number)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def merge_driver_names(df, drivers_df, on="DriverNumber"):
    """Add driver abbreviation and team name to a laps/position dataframe."""
    if drivers_df.empty:
        return df
    drivers_df = drivers_df.rename(columns={
        "driver_number": "DriverNumber",
        "name_acronym":  "Driver",
        "team_name":     "TeamName",
        "team_colour":   "TeamColour",
    })
    drivers_df["DriverNumber"] = drivers_df["DriverNumber"].astype(str)
    df[on] = df[on].astype(str)
    return df.merge(drivers_df[["DriverNumber", "Driver",
                                 "TeamName", "TeamColour"]],
                    on=on, how="left")


def build_lap_features(laps_df, drivers_df):
    """
    Engineer the same features as before so all existing charts work.
    Adds: OvertakeProbability estimate based on pace + tyre age.
    """
    df = merge_driver_names(laps_df, drivers_df)
    df = df.dropna(subset=["LapTimeSeconds"])
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # Rolling pace features
    df["AvgPace3"] = (
        df.groupby("Driver")["LapTimeSeconds"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["LapDelta"] = df.groupby("Driver")["LapTimeSeconds"].diff()
    df["TyreLife"] = pd.to_numeric(df.get("TyreLife", 0),
                                    errors="coerce").fillna(0)

    # Simple overtake probability heuristic
    # (model not available for OpenF1 since it was trained on FastF1 features)
    # Higher probability = fresher tyres + faster recent pace
    df["TyreAdvantage"] = (
        df["TyreLife"] - df.groupby("LapNumber")["TyreLife"]
        .transform("mean")
    )
    # Normalise to 0-100
    df["OvertakeProbability"] = (
        50
        - df["TyreAdvantage"].clip(-10, 10) * 2
        - df["LapDelta"].clip(-2, 2) * 5
    ).clip(0, 100)

    df["Sector1"] = pd.to_numeric(df.get("Sector1", np.nan),
                                   errors="coerce")
    df["Sector2"] = pd.to_numeric(df.get("Sector2", np.nan),
                                   errors="coerce")
    df["Sector3"] = pd.to_numeric(df.get("Sector3", np.nan),
                                   errors="coerce")
    df["Compound"] = df.get("Compound", "UNKNOWN").fillna("UNKNOWN")

    return df
