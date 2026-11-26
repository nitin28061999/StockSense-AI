
import pandas as pd

from src.data import validate


def test_data_quality_audit_reports_correct_counts(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Verify that the data audit correctly identifies quality issues."""

    sample_data = pd.DataFrame(
        [
            {
                "Date": "2026-01-01",
                "Open": 100,
                "High": 110,
                "Low": 90,
                "Close": 105,
                "Volume": 0,
            },
            {
                "Date": "2026-01-02",
                "Open": 105,
                "High": 115,
                "Low": 100,
                "Close": 110,
                "Volume": 100,
            },
            {
                "Date": "2026-01-02",
                "Open": 105,
                "High": 115,
                "Low": 100,
                "Close": 110,
                "Volume": 100,
            },
            {
                "Date": "2026-01-03",
                "Open": 110,
                "High": 105,
                "Low": 108,
                "Close": 109,
                "Volume": 0,
            },
        ]
    )

    csv_path = tmp_path / "sample_nifty50.csv"
    sample_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(validate, "DATA_FILE", csv_path)

    validate.main()

    output = capsys.readouterr().out

    assert "(4, 6)" in output
    assert "Duplicate records:" in output
    assert "1" in output
    assert "Duplicate dates:" in output
    assert "Zero-volume records: 2" in output
    assert "Zero-volume percentage: 50.00%" in output
    assert "Invalid OHLC records: 1" in output
    assert "2026-01-01" in output
    assert "2026-01-03" in output
    assert "AUDIT COMPLETE" in output


def test_data_quality_audit_handles_clean_data(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Verify that clean data produces zero quality issues."""

    sample_data = pd.DataFrame(
        [
            {
                "Date": "2026-01-01",
                "Open": 100,
                "High": 110,
                "Low": 90,
                "Close": 105,
                "Volume": 100,
            },
            {
                "Date": "2026-01-02",
                "Open": 105,
                "High": 115,
                "Low": 100,
                "Close": 110,
                "Volume": 200,
            },
            {
                "Date": "2026-01-03",
                "Open": 110,
                "High": 120,
                "Low": 105,
                "Close": 115,
                "Volume": 300,
            },
        ]
    )

    csv_path = tmp_path / "clean_nifty50.csv"
    sample_data.to_csv(csv_path, index=False)

    monkeypatch.setattr(validate, "DATA_FILE", csv_path)

    validate.main()

    output = capsys.readouterr().out

    assert "Duplicate records:" in output
    assert "Duplicate dates:" in output
    assert "Zero-volume records: 0" in output
    assert "Invalid OHLC records: 0" in output
    assert "AUDIT COMPLETE" in output
