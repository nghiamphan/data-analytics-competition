import cloudscraper
import csv
import re
import json
import os
import pandas as pd
import time

MAX_ATTEMPTS = 5
SLEEP_TIME = [0, 1, 2, 3, 4, 5]

# Cities in Atlantic Canada
URL_HALIFAX = "https://rentals.ca/halifax/all-apartments-condos"  # Halifax, NS
URL_DARTMOUTH = "https://rentals.ca/dartmouth/all-apartments-condos"  # Dartmouth, NS

URL_MONCTON = "https://rentals.ca/moncton/all-apartments-condos"  # Moncton, New Brunswick
URL_SAINT_JOHN = "https://rentals.ca/saint-john/all-apartments-condos"  # Saint John, New Brunswick
URL_FREDERICTON = "https://rentals.ca/fredericton/all-apartments-condos"  # Fredericton, New Brunswick

URL_CHARLOTTETOWN = "https://rentals.ca/charlottetown/all-apartments-condos"  # Charlottetown, PEI

URL_ST_JOHN = "https://rentals.ca/st-johns/all-apartments-condos"  # St. John's, NL

# Choose cities that have somewhat similar rent to Halifax based on this report: https://rentals.ca/national-rent-report
URL_VICTORIA = "https://rentals.ca/victoria/all-apartments-condos"  # Victoria, British Columbia
URL_SURREY = "https://rentals.ca/surrey/all-apartments-condos"  # Surrey, British Columbia (Vanouver)
URL_KELWONA = "https://rentals.ca/kelowna/all-apartments-condos"  # Kelowna, British Columbia

URL_BURLINGTON = "https://rentals.ca/burlington/all-apartments-condos"  # Burlington, Ontario (Toronto)
URL_BRAMPTON = "https://rentals.ca/brampton/all-apartments-condos"  # Brampton, Ontario (Toronto)
URL_HAMILTON = "https://rentals.ca/hamilton/all-apartments-condos"  # Hamilton, Ontario (Toronto)

URL_WATERLOO = "https://rentals.ca/waterloo/all-apartments-condos"  # Waterloo, Ontario (next to Kitchener)
URL_KITCHENER = "https://rentals.ca/kitchener/all-apartments-condos"  # Kitchener, Ontario (next to Waterloo)

URL_GELPH = "https://rentals.ca/guelph/all-apartments-condos"  # Guelph, Ontario (between Kitchener and Toronto)

URL_BARRIE = "https://rentals.ca/barrie/all-apartments-condos"  # Barrie, Ontario (north of Toronto)

URL_OTTAWA = "https://rentals.ca/ottawa/all-apartments-condos"  # Ottawa, Ontario

URL_KINGSTON = "https://rentals.ca/kingston/all-apartments-condos"  # Kingston, Ontario

URL_LONDON = "https://rentals.ca/london/all-apartments-condos"  # London, Ontario

ATLANTIC_CANADA = [
    URL_HALIFAX,
    URL_DARTMOUTH,
    URL_MONCTON,
    URL_SAINT_JOHN,
    URL_FREDERICTON,
    URL_CHARLOTTETOWN,
    URL_ST_JOHN,
]

SIMILAR_RENT_CITIES = [
    URL_VICTORIA,
    URL_SURREY,
    URL_KELWONA,
    URL_BURLINGTON,
    URL_BRAMPTON,
    URL_HAMILTON,
    URL_WATERLOO,
    URL_KITCHENER,
    URL_GELPH,
    URL_BARRIE,
    URL_OTTAWA,
    URL_KINGSTON,
    URL_LONDON,
]

ATTRIBUTES = [
    "building_id",
    "unit_id",
    "company",
    "building_name",
    "address",
    "postal_code",
    "city",
    "property_type",
    "url",
    "pet_friendly",
    "furnished",
    "unit_name",
    "beds",
    "baths",
    "area",
    "rent",
]

AMENITIES = [
    "Pet Friendly",
    "Furnished",
    "Gym",
    "Bike Room",
    "Exercise Room",
    "Fitness Area",
    "Swimming Pool",
    "Recreation Room",
    "Recreation",
    "Heating",
    "Water",
    "Internet / WiFi",
    "Ensuite Laundry",
    "Washer",
    "Laundry Facilities",
    "Parking",
    "Parking - Underground",
]

OUTPUT_CSV_FILE = "./data/units_full_info.csv"

failed_urls = []

scraper = cloudscraper.create_scraper()

existing_unit_ids = set()


def fetch_building_details(url: str) -> json:
    """
    Fetch building details from the given URL.

    Parameters
    ----------
    url : str
        The URL of the building. E.g.: "https://rentals.ca/halifax/200-willett-street"

    Returns
    -------
    data : json
        A JSON object containing the building details.
    """
    attempt = 0

    while attempt < MAX_ATTEMPTS:
        time.sleep(SLEEP_TIME[attempt])
        response = scraper.get(url)

        attempt += 1

        if response.status_code == 200:
            print(f"Fetching successfully: {url}", end="\r")
            # Match the content of the JSON object App.store.listing
            pattern = r"App.store.listing = (\{.*?\});"
            matches = re.findall(pattern, response.text, re.DOTALL)

            if matches:
                return json.loads(matches[0])

            break
        else:
            print(
                f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt}/{MAX_ATTEMPTS}"
            )
            if attempt == MAX_ATTEMPTS:
                failed_urls.append(url)


def extract_building_urls(html_content: str, writer: object, write_to_json: bool = False) -> list:
    """
    Extract building URLs from the given html content.
    Fetch the content from each building URL, which contains the building details and its available units.
    For each available unit, write the unit's details to a CSV file.
    If write_to_json is True, aggregate all building's details as a list and return the list, which will be written in a json file later.
    If write_to_json is False, return an empty list.

    Parameters
    ----------
    html_content : str
        The HTML content of the page.
    writer : csv.writer
        The CSV writer object.
    write_to_json : bool
        If True, aggregate all building's details as a list and return the list.


    Returns
    -------
    buildings : list
        A list of building details.
    """
    buildings = []

    # Match the content within the <script type="application/ld+json"> block
    pattern = r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>'
    matches = re.findall(pattern, html_content, re.DOTALL)

    for match in matches:
        building = json.loads(match)
        url = building.get("url")

        if url:
            data = fetch_building_details(url)
            if data:
                write_row_to_csv(writer, data)

                if write_to_json:
                    buildings.append(data)

    return buildings


def fetch_main_page(main_page_url: str, writer: object, write_to_json: bool = False) -> list:
    """
    Fetch the main page and extract the number of returned pages when search for rentals at certain location, eg.: "https://rentals.ca/halifax/all-apartments-condos".
    For each page, fetch its content and extract building URLs by calling function extract_building_urls(html_content, writer, write_to_json).
    Within the function extract_building_urls, write the unit's detail of the fetched building data into a csv file.
    If write_to_json is True, aggregate all building's details as a list and return the list, which will be written in a json file later.
    If write_to_json is False, return an empty list.s

    Parameters
    ----------
    main_page_url : str
        The URL of the main page, eg.: "https://rentals.ca/halifax/all-apartments-condos"
    writer : csv.writer
        The CSV writer object.
    write_to_json : bool
        If True, aggregate all building's details as a list and return the list.

    Returns
    -------
    buildings : list
        A list of building details.
    """
    attempt = 0
    buildings = []

    while attempt < MAX_ATTEMPTS:
        time.sleep(SLEEP_TIME[attempt])
        response = scraper.get(main_page_url)

        attempt += 1

        if response.status_code == 200:
            print(f"Fetching successfully: {main_page_url}", end="\r")
            # find number of returned pages
            pattern = r'"last_page":(\d+),'
            matches = re.findall(pattern, response.text)
            if matches:
                num_pages = matches[0]
            else:
                num_pages = 1

            # Extract building urls from the first page
            buildings += extract_building_urls(response.text, writer, write_to_json)

            # Extract building urls from the subsequent pages
            for page in range(2, int(num_pages) + 1):

                url = f"{main_page_url}?p={page}"

                attempt_inner = 0
                while attempt_inner < MAX_ATTEMPTS:
                    time.sleep(SLEEP_TIME[attempt_inner])
                    response = scraper.get(url)

                    attempt_inner += 1

                    if response.status_code == 200:
                        print(f"Fetching successfully: {url}", end="\r")
                        buildings += extract_building_urls(response.text, writer, write_to_json)
                        break
                    else:
                        print(
                            f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt_inner}/{MAX_ATTEMPTS}"
                        )
                        if attempt_inner == MAX_ATTEMPTS:
                            failed_urls.append(url)

            break

        else:
            print(
                f"Status code: {response.status_code} - Failed to fetch content from {main_page_url} - Attempt {attempt}/{MAX_ATTEMPTS}"
            )
            if attempt == MAX_ATTEMPTS:
                failed_urls.append(main_page_url)

    return buildings


def write_row_to_csv(writer: object, building_data: json):
    company = building_data.get("company")
    if company:
        company_name = company.get("name")
    else:
        company_name = None

    for unit in building_data.get("units"):
        # Skip if the unit already exists in the CSV file
        if unit.get("id") in existing_unit_ids:
            continue

        row = [
            building_data.get("id"),
            unit.get("id"),
            company_name,
            building_data.get("name"),
            building_data.get("address1"),
            building_data.get("postal_code"),
            building_data.get("city_name"),
            building_data.get("property_type"),
            building_data.get("url"),
            building_data.get("pet_friendly"),
            building_data.get("furnished"),
            unit.get("name"),
            unit.get("beds"),
            unit.get("baths"),
            unit.get("dimensions"),
            unit.get("rent"),
        ]

        # For each amenity, add True if the amenity is in the building's amenities, else add an empty string
        for amenity in AMENITIES:
            row.append(True if amenity in [a.get("name") for a in building_data.get("amenities")] else "")

        writer.writerow(row)


def data_pipeline(
    fetch_data: bool = False,
    main_urls: list[str] = [],
    output_csv_file: str = None,
    output_json_file: str = None,
):
    """
    If fetch_data is True and main_urls is not None and output_csv_file is not None, write the fetched data into the csv file.
    If output_json_file is also not None, will write the fetched data into the json file.

    Parameters
    ----------
    fetch_data : bool
        If True, fetch data from the a list of urls.
    main_urls : list[str]
        A list of main urls. E.g.: ["https://rentals.ca/halifax/all-apartments-condos", "https://rentals.ca/dartmouth/all-apartments-condos"]
    output_csv_file : str
        The path to the output CSV file, which contains the unit details.
    output_json_file : str
        The path to the output JSON file, which contains the unit details.
    """
    global existing_unit_ids

    if not main_urls or not fetch_data or not output_csv_file:
        return

    if not os.path.exists("data"):
        os.makedirs("data")

    with open(output_csv_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header to the CSV file if the file is empty
        if os.path.getsize(output_csv_file) == 0:
            header = ATTRIBUTES + AMENITIES
            writer.writerow(header)

        # Read the existing unit ids from the CSV file
        df = pd.read_csv(output_csv_file)
        existing_unit_ids = set(df["unit_id"])

        buildings = []
        for main_url in main_urls:
            buildings += fetch_main_page(main_url, writer, write_to_json=(output_json_file is not None))

        if output_json_file:
            with open(output_json_file, "w", encoding="utf-8") as file:
                file.write(json.dumps(buildings, indent=2))

        if failed_urls:
            print("\n\nFailed URLs:")
            for url in failed_urls:
                print(url)
        else:
            print("\n\nAll URLs fetched successfully!")


def main():
    data_pipeline(
        fetch_data=True,
        main_urls=ATLANTIC_CANADA + SIMILAR_RENT_CITIES,
        output_csv_file=OUTPUT_CSV_FILE,
    )


if __name__ == "__main__":
    main()
