# Module downloads 1 month of observations from PeMS stiched into one csv.gz file
# Returns the path to the output file
#
# Running from the terminal example:
# python load_pems_data.py --year 2018 --month 4 --district 7 --highways [10, 605]
#
# Can run python load_pems_data.py --help to see summary of arguments
# 
# --year: 4-digit calendar year
# --month: 1-12
# --district: 1-12
# --highways: (optional) list of freeway numbers (int) to include; None includes all
# --lanes: (optional) include this if individual lane data is to be included
# --dow: (optional) days of week, options are MTWRFSU (R = Thurs, U = Sun), e.g. --dow SU is weekends; omitting includes all
# --filedir: (optional) directory to save the file to; default is ../../data/


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from getpass import getpass
from playwright.async_api import async_playwright
import requests
import io
import json
import asyncio

from pathlib import Path
import argparse
import gzip
import tempfile

from datetime import date
from datetime import datetime
import calendar


DATA_TYPE = "station_5min"
# The column names for the 5min station data type
COLUMNS = [
    "Timestamp",
    "Station",
    "District",
    "Freeway #",
    "Direction of Travel",
    "Lane Type",
    "Station Length",
    "Samples",
    "% Observed",
    "Total Flow",
    "Avg Occupancy",
    "Avg Speed",
    "Lane 1 Samples",
    "Lane 1 Flow",
    "Lane 1 Avg Occ",
    "Lane 1 Avg Speed",
    "Lane 1 Observed",
    "Lane 2 Samples",
    "Lane 2 Flow",
    "Lane 2 Avg Occ",
    "Lane 2 Avg Speed",
    "Lane 2 Observed",
    "Lane 3 Samples",
    "Lane 3 Flow",
    "Lane 3 Avg Occ",
    "Lane 3 Avg Speed",
    "Lane 3 Observed",
    "Lane 4 Samples",
    "Lane 4 Flow",
    "Lane 4 Avg Occ",
    "Lane 4 Avg Speed",
    "Lane 4 Observed",
    "Lane 5 Samples",
    "Lane 5 Flow",
    "Lane 5 Avg Occ",
    "Lane 5 Avg Speed",
    "Lane 5 Observed",
    "Lane 6 Samples",
    "Lane 6 Flow",
    "Lane 6 Avg Occ",
    "Lane 6 Avg Speed",
    "Lane 6 Observed",
    "Lane 7 Samples",
    "Lane 7 Flow",
    "Lane 7 Avg Occ",
    "Lane 7 Avg Speed",
    "Lane 7 Observed",
    "Lane 8 Samples",
    "Lane 8 Flow",
    "Lane 8 Avg Occ",
    "Lane 8 Avg Speed",
    "Lane 8 Observed",
]

# Parameters
# year 4 digit
# month 1-12
# district 1-12
# highways numbered (keep all if None)
# lanes: keep individual lane data if true
# dow: day of the week, MTWRFSU (R = Thurs, U = Sun)
# filedir: custom directory if desired
async def fetch_month(year, month, district, highways=None, lanes=False, dow=None, filedir='./data/'):

    # Date range
    start_date = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_date = date(year, month, last_day)

    # If lane data not requested, do not include it
    columns = COLUMNS if lanes else COLUMNS[:12]

    read_columns = list(dict.fromkeys(columns + ["Freeway #"]))

    # Output csv path
    output_dir = Path(filedir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"d{district:02d}_{year}_{month:02d}_station_5min.csv.gz"
    )

    # Credentials
    username = input("PeMS username: ")
    password = getpass("PeMS password: ")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        page.set_default_timeout(60_000)

        try:
            await page.goto(
                "https://pems.dot.ca.gov/?dnode=Clearinghouse",
                wait_until="domcontentloaded",
                timeout=120_000,
            )

            # Authenticate with credentials
            await page.locator("#username").fill(username)
            await page.locator("#password").fill(password)
            await page.locator('[name="login"]').click()

            # Wait for the authenticated Clearinghouse interface
            await page.locator("#district_id").wait_for(state="visible")

            print("Login successful. Data Clearinghouse is open.")

            # Select data type
            await page.locator("#type").select_option(DATA_TYPE)
            await page.locator(
                f'#district_id option[value="{district}"]'
            ).wait_for(state="attached")
            await page.locator("#district_id").select_option(str(district))
            await page.locator('[name="submit"]').click()

            # Get the link names
            archives = {}

            # Pick out the year in the widget table
            year_cell = page.locator("td.widgetYear").filter(
                has_text=re.compile(rf"^\s*{year % 100:02d}\s*$")
            )

            # Month cells indexed from zero, click on the appropriate month cell
            await year_cell.locator(
                "xpath=following-sibling::td"
            ).nth(month - 1).click()

            # File naming convention
            prefix = f"d{district:02d}_text_{DATA_TYPE}_{year}_{month:02d}"

            # Wait for archive links to load
            await page.locator("a[href]").filter(
                has_text=re.compile(rf"^{re.escape(prefix)}")
            ).first.wait_for(state="attached")

            # Get full URL
            # done via JavaScript to avoid performing a new browser operation for each link
            links = await page.locator("a[href]").evaluate_all("""
                elements => elements.map(a => ({
                    filename: a.textContent.trim(),
                    url: a.href
                }))    
            """)

            for link in links:
                if DATA_TYPE in link["filename"]:
                    print(repr(link["filename"]), link["url"])

            for link in links:
                filename = link["filename"]

                if not (
                    filename.startswith(prefix)
                    and filename.endswith(".txt.gz")
                ):
                    continue

                date_text = filename.removesuffix(".txt.gz")[-10:]
                file_date = date.fromisoformat(date_text.replace("_", "-"))

                if start_date <= file_date <= end_date:
                    archives[filename] = link["url"]

            print(f"Found {len(archives)} file(s) for the given dates.")

            # Move to requests to download the files
            with requests.Session() as session:
                session.headers["User-Agent"] = await page.evaluate(
                    "navigator.userAgent"
                )

                # Copy cookies from Playwright
                for cookie in await context.cookies(
                    "https://pems.dot.ca.gov/"
                ):
                    session.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie["domain"],
                        path=cookie["path"],
                        secure=cookie["secure"],
                    )

                with gzip.open(output_path, "wt", newline="") as monthly_output:
                    # First write the column names
                    pd.DataFrame(columns=columns).to_csv(monthly_output, index=False)

                    for filename, url in sorted(archives.items()):
                        file_date = date.fromisoformat(
                            filename.removesuffix(".txt.gz")[-10:].replace("_","-")
                        )

                        # Optional day of the week selection
                        if dow is not None and "MTWRFSU"[file_date.weekday()] not in dow:
                            continue

                        print(f"Processing {filename}", flush=True)

                        # Delete the daily files after processing them into the month
                        with tempfile.TemporaryDirectory() as temporary_dir:
                            archive_path = Path(temporary_dir) / filename

                            with session.get(
                                url, stream=True, timeout=(15,120)
                            ) as response:
                                response.raise_for_status()

                                # Download by blocks
                                with archive_path.open("wb") as downloaded:
                                    for block in response.iter_content(1024*1024):
                                        if block:
                                            downloaded.write(block)

                            # Read into memory in chunks
                            for chunk in pd.read_csv(
                                archive_path,
                                header=None,
                                names=COLUMNS,
                                usecols=read_columns,
                                chunksize=100_000,
                            ):
                                # Select the given highways if present
                                if highways is not None:
                                    chunk = chunk.loc[chunk["Freeway #"].isin(highways)]

                                chunk.loc[:, columns].to_csv(monthly_output, index=False, header=False)

            print(f"Wrote {output_path}.")

        finally:
            # Close the browser even if a selection or download fails
            await context.close()
            await browser.close()

        return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Download PeMS station 5-minute data for one district and month, filter it and save to a csv.gz file."
    )

    parser.add_argument("--year", type=int, required=True, help="Year of observations, e.g. 2026")
    parser.add_argument("--month", type=int, required=True, help="Month of observations, 1 through 12")
    parser.add_argument("--district", type=int, required=True, help="Caltrans district number, 1 through 12")
    parser.add_argument("--highways", type=int, nargs="+", default=None, help="Space-separated list of highways to retain, e.g. 10 605; omit to retain all highways")
    parser.add_argument("--lanes", action="store_true", help="Include individual lane columns")
    parser.add_argument("--dow", default=None, help="Days of the week to retain: MTWRFSU; omit for all days")
    parser.add_argument("--filedir", default="../../data/", help="Output directory; default ../../data/")

    args = parser.parse_args()

    if args.dow is not None:
        args.dow = args.dow.upper()
        if not args.dow or any(day not in "MTWRFSU" for day in args.dow):
            parser.error("--dow must contain only letters from MTWRFSU")

    s = datetime.now()

    output_path = asyncio.run(
        fetch_month(
            year=args.year,
            month=args.month,
            district=args.district,
            highways=args.highways,
            lanes=args.lanes,
            dow=args.dow,
            filedir=args.filedir,
        )
    )

    e = datetime.now()
    print(f"Time elapsed: {(e-s).total_seconds()} seconds")

if __name__ == "__main__":
    main()