import scraping
from pathlib import Path

if __name__ == "__main__":
    naikaku_source_path = Path("source/holiday_naikaku.csv")
    scraping.make_source_with_naikaku(naikaku_source_path)
    api_source_path = Path("source/holiday_api.csv")
    scraping.make_source_with_api(api_source_path)
    jpholiday_source_path = Path("source/holiday_jpholiday.csv")
    scraping.make_source_with_jpholiday(jpholiday_source_path)