import requests
import re
import json
import os
import time

HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_TIME = [1, 2, 3, 4, 5]

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
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
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
                f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt}/{max_attempts}"
            )
            if attempt == max_attempts:
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


def fetch_main_page(main_page_url):
    """
    Fetch the main page and extract the number of pages.
    Then fetch building URLs from each page.
    Finally, aggregate the data from each building URL and save it to a JSON file.

    Parameters
    ----------
    main_page_url : str
        The URL of the main page, eg.: "https://rentals.ca/halifax/all-apartments-condos"
    """
    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
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
                while attempt_inner < max_attempts:
                    time.sleep(SLEEP_TIME[attempt_inner])
                    response = requests.get(url, headers=HEADERS)

                    attempt_inner += 1

                    if response.status_code == 200:
                        buildings += extract_building_urls(response.text)
                        break
                    else:
                        print(
                            f"Status code: {response.status_code} - Failed to fetch content from {url} - Attempt {attempt_inner}/{max_attempts}"
                        )
                        if attempt_inner == max_attempts:
                            failed_urls.append(url)

            with open("./data/buildings.json", "w", encoding="utf-8") as file:
                file.write(json.dumps(buildings, indent=2))

            break

        else:
            print(
                f"Status code: {response.status_code} - Failed to fetch content from {main_page_url} - Attempt {attempt}/{max_attempts}"
            )
            if attempt == max_attempts:
                failed_urls.append(main_page_url)


def main():
    URL_APARTMENTS_CONDOS = "https://rentals.ca/halifax/all-apartments-condos"
    URL_APARTMENTS = "https://rentals.ca/halifax/all-apartments"
    URL_CONDOS = "https://rentals.ca/halifax/all-condos"

    if not os.path.exists("data"):
        os.makedirs("data")

    fetch_main_page(URL_APARTMENTS_CONDOS)

    if failed_urls:
        print("\nFailed URLs:")
        for url in failed_urls:
            print(url)


if __name__ == "__main__":
    main()
