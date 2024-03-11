import csv
import requests
import re
import json
import os
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
MAX_ATTEMPTS = 5
SLEEP_TIME = [1, 2, 3, 4, 5]

URL_APARTMENTS_CONDOS = "https://rentals.ca/halifax/all-apartments-condos"
URL_APARTMENTS = "https://rentals.ca/halifax/all-apartments"
URL_CONDOS = "https://rentals.ca/halifax/all-condos"

failed_urls = []


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
        response = requests.get(url, headers=HEADERS)

        attempt += 1

        if response.status_code == 200:
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


def fetch_main_page(main_page_url, json_file):
    """
    Fetch the main page and extract the number of pages.
    Then fetch building URLs from each page.
    Finally, aggregate the data from each building URL and save it to a JSON file.

    Parameters
    ----------
    main_page_url : str
        The URL of the main page, eg.: "https://rentals.ca/halifax/all-apartments-condos"
    json_file : str
        The path to the JSON file, which contains the building details.
    """
    attempt = 0

    while attempt < MAX_ATTEMPTS:
        time.sleep(SLEEP_TIME[attempt])
        response = requests.get(main_page_url, headers=HEADERS)

        attempt += 1

        if response.status_code == 200:
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
                    response = requests.get(url, headers=HEADERS)

                    attempt_inner += 1

                    if response.status_code == 200:
                        buildings += extract_building_urls(response.text)
                        break
                    else:
                        print(
                            f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt_inner}/{MAX_ATTEMPTS}"
                        )
                        if attempt_inner == MAX_ATTEMPTS:
                            failed_urls.append(url)

            with open(json_file, "w", encoding="utf-8") as file:
                file.write(json.dumps(buildings, indent=2))

            break

        else:
            print(
                f"Status code: {response.status_code} - Failed to fetch content from {main_page_url} - Attempt {attempt}/{MAX_ATTEMPTS}"
            )
            if attempt == MAX_ATTEMPTS:
                failed_urls.append(main_page_url)


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


def write_data_to_two_csv(json_file, building_csv_file, unit_csv_file):
    """
    Write the data from the JSON file to two CSV files: buildings.csv and units.csv

    Parameters
    ----------
    json_file : str
        The path to the JSON file, which contains the building details.
    building_csv_file : str
        The path to the building CSV file, which contains information of the buildings.
    unit_csv_file : str
        The path to the unit CSV file, which contains information of the available units in the buildings.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        buildings = json.load(file)

    with open(building_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = [
            "id",
            "company",
            "name",
            "address",
            "postal_code",
            "city",
            "property_type",
            "url",
            "view_on_map_url",
            "pet_friendly",
            "furnished",
            "amenities",
        ]
        writer.writerow(header)

        for building in buildings:
            company = building.get("company")
            if company:
                company_name = company.get("name")
            else:
                company_name = None

            row = [
                building.get("id"),
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

            for amenity in building.get("amenities"):
                row.append(amenity.get("name"))

            writer.writerow(row)

    with open(unit_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = ["building_id", "unit_id", "name", "beds", "baths", "area", "rent"]
        writer.writerow(header)

        for building in buildings:
            for unit in building.get("units"):
                row = [
                    building.get("id"),
                    unit.get("id"),
                    unit.get("name"),
                    unit.get("beds"),
                    unit.get("baths"),
                    unit.get("dimensions"),
                    unit.get("rent"),
                ]

                writer.writerow(row)


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
    main_url: str = None,
    json_file: str = None,
    building_csv_file: str = None,
    unit_csv_file: str = None,
    unit_full_info_csv_file: str = None,
):
    """
    If fetch_data is True and main_url is not None and json_file is not None,
    call function fetch_main_page(main_url, json_file) that will write fetched data into the json file.

    If json_file is not None and building_csv_file is not None and unit_csv_file is not None,
    call function write_data_to_two_csv(json_file, building_csv_file, unit_csv_file) that will write data from the json file into two csv files.

    If json_file is not None and unit_full_info_csv_file is not None,
    call function write_data_to_one_csv(json_file, unit_full_info_csv_file) that will write data from the json file into one csv file.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    if fetch_data and main_url and json_file:
        fetch_main_page(main_url, json_file)

        if failed_urls:
            print("\nFailed URLs:")
            for url in failed_urls:
                print(url)
        else:
            print("\nAll URLs fetched successfully!")

    if json_file and building_csv_file and unit_csv_file:
        write_data_to_two_csv(json_file, building_csv_file, unit_csv_file)

    if json_file and unit_full_info_csv_file:
        write_data_to_one_csv(json_file, unit_full_info_csv_file)


def main():
    data_pipeline(
        fetch_data=False,
        main_url=URL_APARTMENTS_CONDOS,
        json_file="./data/buildings.json",
        unit_full_info_csv_file="./data/units_full_info.csv",
    )


if __name__ == "__main__":
    main()
