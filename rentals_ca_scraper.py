import cloudscraper
import csv
import re
import json
import os
import time

MAX_ATTEMPTS = 5
SLEEP_TIME = [0, 1, 2, 3, 4, 5]

URL_APARTMENTS_CONDOS_HALIFAX = "https://rentals.ca/halifax/all-apartments-condos"
URL_APARTMENTS_HALIFAX = "https://rentals.ca/halifax/all-apartments"
URL_CONDOS_HALIFAX = "https://rentals.ca/halifax/all-condos"

URL_APARTMENTS_CONDOS_DARTMOUTH = "https://rentals.ca/dartmouth/all-apartments-condos"

URL_APARTMENTS_CONDOS_SAINT_JOHN = "https://rentals.ca/saint-john/all-apartments-condos"  # Saint John, New Brunswick

URL_APARTMENTS_CONDOS_FREDERICTON = "https://rentals.ca/fredericton/all-apartments-condos"  # Fredericton, New Brunswick

URL_APARTMENTS_CONDOS_MONCTON = "https://rentals.ca/moncton/all-apartments-condos"  # Moncton, New Brunswick

URL_APARTMENTS_CONDOS_CHARLOTTETOWN = "https://rentals.ca/charlottetown/all-apartments-condos"  # Charlottetown, PEI

URL_APARTMENTS_CONDOS_ST_JOHN = "https://rentals.ca/st-johns/all-apartments-condos"  # St. John's, NL

URL_APARTMENTS_CONDOS_WINNIPEG = "https://rentals.ca/winnipeg/all-apartments-condos"  # Winnipeg, Manitoba

URL_APARTMENTS_CONDOS_REGINA = "https://rentals.ca/regina/all-apartments-condos"  # Regina, Saskatchewan

URL_APARTMENTS_CONDOS_SASKATOON = "https://rentals.ca/saskatoon/all-apartments-condos"  # Saskatoon, Saskatchewan

URL_APARTMENTS_CONDOS_EDMONTON = "https://rentals.ca/edmonton/all-apartments-condos"  # Edmonton, Alberta

URL_APARTMENTS_CONDOS_CALGARY = "https://rentals.ca/calgary/all-apartments-condos"  # Calgary, Alberta

URL_APARTMENTS_CONDOS_HAMILTON = "https://rentals.ca/hamilton/all-apartments-condos"  # Hamilton, Ontario

URL_APARTMENTS_CONDOS_MISSISSAUGA = "https://rentals.ca/mississauga/all-apartments-condos"  # Mississauga, Ontario

URL_APARTMENTS_CONDOS_TORONTO = "https://rentals.ca/toronto/all-apartments-condos"  # Toronto, Ontario

failed_urls = []

scraper = cloudscraper.create_scraper()


def fetch_building_details(url) -> json:
    """
    Fetch building details from the given URL

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


def extract_building_urls(text) -> list:
    """
    Extract building URLs from the given text.
    Then fetch building details from each URL and aggregate all building's details as a list.

    Parameters
    ----------
    text : str
        The HTML content of the page.

    Returns
    -------
    buildings : list
        A list of building details.
    """
    buildings = []

    # Match the content within the <script type="application/ld+json"> block
    pattern = r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        building = json.loads(match)
        url = building.get("url")

        if url:
            data = fetch_building_details(url)
            if data:
                buildings.append(data)

    return buildings


def fetch_main_page(main_page_url) -> list:
    """
    Fetch the main page and extract the number of pages.
    Then fetch building URLs from each page.
    Return the aggregate of data from each building URL.

    Parameters
    ----------
    main_page_url : str
        The URL of the main page, eg.: "https://rentals.ca/halifax/all-apartments-condos"

    Returns
    -------
    buildings : list
        A list of building details.
    """
    attempt = 0

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
            buildings = extract_building_urls(response.text)

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
                        buildings += extract_building_urls(response.text)
                        break
                    else:
                        print(
                            f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt_inner}/{MAX_ATTEMPTS}"
                        )
                        if attempt_inner == MAX_ATTEMPTS:
                            failed_urls.append(url)

            return buildings

        else:
            print(
                f"Status code: {response.status_code} - Failed to fetch content from {main_page_url} - Attempt {attempt}/{MAX_ATTEMPTS}"
            )
            if attempt == MAX_ATTEMPTS:
                failed_urls.append(main_page_url)

    return []


def get_amenities(json_file) -> list:
    """
    Get all amenities from the JSON file.

    Parameters
    ----------
    json_file : str
        The path to the JSON file, which contains the building details.

    Returns
    -------
    amenities : list
        A list of all distinct amenities of all buildings.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        buildings = json.load(file)

    amenities = set()

    for building in buildings:
        for amenity in building.get("amenities"):
            amenities.add(amenity.get("name"))

    return list(amenities)


def write_data_to_one_csv(json_file, unit_full_info_csv_file):
    """
    Write the data from the JSON file to one CSV file: units.csv

    Parameters
    ----------
    json_file : str
        The path to the JSON file, which contains the building details.
    unit_full_info_csv_file : str
        The path to the unit CSV file, each row contains the information of a unit and of the building the unit belongs to.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        buildings = json.load(file)

    with open(unit_full_info_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = [
            "building_id",
            "unit_id",
            "company",
            "building_name",
            "address",
            "postal_code",
            "city",
            "property_type",
            "url",
            "view_on_map_url",
            "pet_friendly",
            "furnished",
        ]

        amenities = get_amenities(json_file)  # list of all possible amenities
        header += amenities

        header += ["unit_name", "beds", "baths", "area", "rent"]

        writer.writerow(header)

        for building in buildings:
            company = building.get("company")
            if company:
                company_name = company.get("name")
            else:
                company_name = None

            for unit in building.get("units"):
                row = [
                    building.get("id"),
                    unit.get("id"),
                    company_name,
                    building.get("name"),
                    building.get("address1"),
                    building.get("postal_code"),
                    building.get("city_name"),
                    building.get("property_type"),
                    building.get("url"),
                    building.get("view_on_map_url"),
                    building.get("pet_friendly"),
                    building.get("furnished"),
                ]

                # For each amenity, add True if the amenity is in the building's amenities, else add an empty string
                for amenity in amenities:
                    row.append(True if amenity in [a.get("name") for a in building.get("amenities")] else "")

                row += [
                    unit.get("name"),
                    unit.get("beds"),
                    unit.get("baths"),
                    unit.get("dimensions"),
                    unit.get("rent"),
                ]

                writer.writerow(row)


def data_pipeline(
    fetch_data: bool = False,
    main_urls: list[str] = [],
    json_file: str = None,
    unit_full_info_csv_file: str = None,
):
    """
    If fetch_data is True and main_url is not None and json_file is not None,
    call function fetch_main_page(main_url, json_file) that will write fetched data into the json file.

    If json_file is not None and unit_full_info_csv_file is not None,
    call function write_data_to_one_csv(json_file, unit_full_info_csv_file) that will write data from the json file into one csv file.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    buildings = []
    if fetch_data and len(main_urls) > 0 and json_file:
        for main_url in main_urls:
            buildings += fetch_main_page(main_url)

        with open(json_file, "w", encoding="utf-8") as file:
            file.write(json.dumps(buildings, indent=2))

        if failed_urls:
            print("\nFailed URLs:")
            for url in failed_urls:
                print(url)
        else:
            print("\nAll URLs fetched successfully!")

    if json_file and unit_full_info_csv_file:
        write_data_to_one_csv(json_file, unit_full_info_csv_file)


def main():
    main_urls = [
        URL_APARTMENTS_CONDOS_HALIFAX,
        URL_APARTMENTS_CONDOS_DARTMOUTH,
        URL_APARTMENTS_CONDOS_SAINT_JOHN,
        URL_APARTMENTS_CONDOS_FREDERICTON,
        URL_APARTMENTS_CONDOS_MONCTON,
        URL_APARTMENTS_CONDOS_CHARLOTTETOWN,
        URL_APARTMENTS_CONDOS_ST_JOHN,
        URL_APARTMENTS_CONDOS_WINNIPEG,
        URL_APARTMENTS_CONDOS_REGINA,
        URL_APARTMENTS_CONDOS_SASKATOON,
        URL_APARTMENTS_CONDOS_EDMONTON,
        URL_APARTMENTS_CONDOS_CALGARY,
        URL_APARTMENTS_CONDOS_HAMILTON,
        URL_APARTMENTS_CONDOS_MISSISSAUGA,
    ]

    data_pipeline(
        fetch_data=True,
        main_urls=main_urls,
        json_file="./data/buildings.json",
        unit_full_info_csv_file="./data/units_full_info.csv",
    )


if __name__ == "__main__":
    main()
