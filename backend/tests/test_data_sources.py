from backend.data_sources.csv_source import CsvMatchResultSource


def test_csv_source_loads_and_sorts_by_date(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2020-01-05,B,A,1,0,Friendly,X,X,FALSE\n"
        "2019-01-01,A,B,2,1,Friendly,X,X,FALSE\n"
    )

    df = CsvMatchResultSource(csv_path).load()

    assert list(df["home_team"]) == ["A", "B"]
    assert df["date"].is_monotonic_increasing
