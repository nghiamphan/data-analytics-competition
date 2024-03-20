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


failed_urls = []

scraper = cloudscraper.create_scraper()


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
    csv_file: str = None,
    json_file: str = None,
):
    """
    If fetch_data is True and main_urls is not None and csv_file is not None, write the fetched data into a csv file.
    If json_file is also not None, will write the fetched data into a json file.

    Parameters
    ----------
    fetch_data : bool
        If True, fetch data from the a list of urls.
    main_urls : list[str]
        A list of main urls. E.g.: ["https://rentals.ca/halifax/all-apartments-condos", "https://rentals.ca/dartmouth/all-apartments-condos"]
    json_file : str
        The path to the output JSON file, which contains the building details.
    csv_file : str
        The path to the output CSV file, which contains the unit details.
    """
    if not main_urls or not fetch_data or not csv_file:
        return

    if not os.path.exists("data"):
        os.makedirs("data")

    with open(csv_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header to the CSV file if the file is empty
        if os.path.getsize(csv_file) == 0:
            header = ATTRIBUTES + AMENITIES
            writer.writerow(header)

        buildings = []
        for main_url in main_urls:
            buildings += fetch_main_page(main_url, writer, write_to_json=(json_file is not None))

        if json_file:
            with open(json_file, "w", encoding="utf-8") as file:
                file.write(json.dumps(buildings, indent=2))

        if failed_urls:
            print("\nFailed URLs:")
            for url in failed_urls:
                print(url)
        else:
            print("\nAll URLs fetched successfully!")


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
        csv_file="./data/units_full_info.csv",
    )


if __name__ == "__main__":
    main()
