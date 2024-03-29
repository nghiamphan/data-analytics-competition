import cloudscraper
import csv
import re
import json
import os
import pandas as pd
import time

from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 5
SLEEP_TIME = [0, 1, 2, 3, 4, 5]

# Cities in Atlantic Canada
URL_HALIFAX = "https://rentals.ca/halifax/all-apartments-condos"  # Halifax, NS
URL_BEDFORD = "https://rentals.ca/bedford/all-apartments-condos"  # Bedford, NS
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

HALIFAX = [
    URL_HALIFAX,
    URL_BEDFORD,
    URL_DARTMOUTH,
]

ATLANTIC_CANADA = [
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
    "longitude",
    "latitude",
    "property_type",
    "url",
    "pet_friendly",
    "furnished",
    "unit_name",
    "beds",
    "baths",
    "area",
    "rent_to_unit_area_ratio",
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

NEIGHBORHOOD_SCORES = [
    "daycares",
    "primary_schools",
    "high_schools",
    "groceries",
    "shopping",
    "restaurants",
    "cafes",
    "pedestrian_friendly",
    # "cycling_friendly",
    "transit_friendly",
    "car_friendly",
    "vibrant",
    "nightlife",
    "quiet",
    "parks",
    # "greenery",
]

LOCALLOGIC_API_TOKEN = os.getenv("LOCALLOGIC_API_TOKEN")

OUTPUT_CSV_FILE_RAW = "./data/units_info_raw.csv"
OUTPUT_CSV_FILE_PROCESSED = "./data/units_info_processed.csv"
OUTPUT_CSV_FILE_PROCESSED_HALIFAX = "./data/units_info_processed_halifax.csv"

failed_urls = []
n_missing_neighborhood_scores = 0

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
    """
    Write the unit's detail of the fetched building data into a csv file.

    Parameters
    ----------
    writer : csv.writer
        The CSV writer object.
    building_data : json
        The building data.
    """
    global n_missing_neighborhood_scores

    company = building_data.get("company")
    if company:
        company_name = company.get("name")
    else:
        company_name = None

    # Fetch neighborhood scores from LocalLogic API (we will set the score scale 0-1)
    neighborhood_scores = {}
    for key in NEIGHBORHOOD_SCORES:
        neighborhood_scores[key] = 0

    longitude = building_data.get("location").get("lng")
    latitude = building_data.get("location").get("lat")

    locallogic_url = f"https://api.locallogic.co/v1/scores?token={LOCALLOGIC_API_TOKEN}&lng={longitude}&lat={latitude}"
    locallogic_response = scraper.get(locallogic_url)

    if locallogic_response.status_code == 200:
        response_scores = locallogic_response.json().get("data").get("attributes")

        if response_scores:
            for key in response_scores.keys():
                neighborhood_scores[key] = response_scores[key].get("value") / 5
            n_missing_neighborhood_scores += max(0, len(NEIGHBORHOOD_SCORES) - len(response_scores.keys()))
        else:
            n_missing_neighborhood_scores += len(NEIGHBORHOOD_SCORES)

    else:
        print(f"Status code: {locallogic_response.status_code} - Failed to fetch content from {locallogic_url}")
        failed_urls.append(locallogic_url)

    for unit in building_data.get("units"):

        # Calculate rent to unit area ratio
        area = unit.get("dimensions")
        if area:
            rent_to_unit_area_ratio = unit.get("rent") / area
        else:
            rent_to_unit_area_ratio = None

        # create a row for each unit
        row = [
            building_data.get("id"),
            unit.get("id"),
            company_name,
            building_data.get("name"),
            building_data.get("address1"),
            building_data.get("postal_code"),
            building_data.get("city_name"),
            longitude,
            latitude,
            building_data.get("property_type"),
            building_data.get("url"),
            building_data.get("pet_friendly"),
            building_data.get("furnished"),
            unit.get("name"),
            unit.get("beds"),
            unit.get("baths"),
            area,
            rent_to_unit_area_ratio,
            unit.get("rent"),
        ]

        # For each amenity, add True if the amenity is in the building's amenities, else add an empty string
        for amenity in AMENITIES:
            row.append(True if amenity in [a.get("name") for a in building_data.get("amenities")] else "")

        # Add neighborhood scores
        row += [neighborhood_scores[key] for key in NEIGHBORHOOD_SCORES]

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
    if not main_urls or not fetch_data or not output_csv_file:
        return

    if not os.path.exists("data"):
        os.makedirs("data")

    with open(output_csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(ATTRIBUTES + AMENITIES + NEIGHBORHOOD_SCORES)

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

        print(f"\n\nNumber of missing neighborhood scores: {n_missing_neighborhood_scores}")


def process_amenities(csv_file_raw: str, csv_file_processed: str):
    """
    Combine the amenity columns into a single column for each amenity.

    Parameters
    ----------
    csv_file_raw : str
        The path to the raw CSV file.
    csv_file_processed : str
        The path to the processed CSV file where the amenity columns are combined.
    """
    df = pd.read_csv(csv_file_raw)

    # Process the 'pet_friendly' column
    df["pet_friendly"] = df[["pet_friendly", "Pet Friendly"]].any(axis=True).astype(int)

    # Process the 'furnished' column
    df["furnished"] = df[["furnished", "Furnished"]].any(axis=True).astype(int)

    # Process the 'fitness_center' column
    df["fitness_center"] = df[["Gym", "Bike Room", "Exercise Room", "Fitness Area"]].any(axis=True).astype(int)

    # Process the "swimmming_pool" column
    df["swimming_pool"] = df[["Swimming Pool"]].any(axis=True).astype(int)

    # Process the "recreation_room" column
    df["recreation_room"] = df[["Recreation Room", "Recreation"]].any(axis=True).astype(int)

    # Process the "heating" column
    df["heating"] = df[["Heating"]].any(axis=True).astype(int)

    # Process the "water" column
    df["water"] = df[["Water"]].any(axis=True).astype(int)

    # Process the "Internet" column
    df["internet"] = df[["Internet / WiFi"]].any(axis=True).astype(int)

    # Process the "ensuite_laundry" column
    df["ensuite_laundry"] = df[["Ensuite Laundry", "Washer"]].any(axis=True).astype(int)

    # Process the "laundry_room" column
    df["laundry_room"] = df[["Laundry Facilities"]].any(axis=True).astype(int)

    # Process the 'parking' column
    df["parking"] = df[["Parking"]].any(axis=True).astype(int)

    # Process the 'undergrounnd_parking' column
    df["underground_parking"] = df[["Parking - Underground"]].any(axis=True).astype(int)

    df.drop(
        columns=[
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
        ],
        inplace=True,
    )

    # Save the processed data to a new CSV file
    df.to_csv(csv_file_processed, index=False)

    # Save the processed data for Halifax to a new CSV file
    df_halifax = df[df["city"].isin(["Halifax", "Bedford", "Dartmouth"])]
    df_halifax.to_csv(OUTPUT_CSV_FILE_PROCESSED_HALIFAX, index=False)


def main():
    data_pipeline(
        fetch_data=True,
        main_urls=HALIFAX + ATLANTIC_CANADA + SIMILAR_RENT_CITIES,
        output_csv_file=OUTPUT_CSV_FILE_RAW,
    )

    process_amenities(OUTPUT_CSV_FILE_RAW, OUTPUT_CSV_FILE_PROCESSED)


if __name__ == "__main__":
    main()
