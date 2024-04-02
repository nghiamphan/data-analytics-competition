import streamlit as st
import cloudscraper
import pandas as pd
import torch
import joblib
import json
import os
import warnings

from dotenv import load_dotenv
from neural_network_model import (
    NeuralNetwork,
    JSON_FILE_POSTAL_CODE_IDX_MAPPING,
    JSON_FILE_POSTAL_CODE_FIRST_3_IDX_MAPPING,
    PICKLE_FILE_INPUT_SCALER,
    PICKLE_FILE_RENT_SCALER,
    JSON_FILE_BEST_PARAMS,
    FILE_TORCH_MODEL,
)
from rentals_ca_scraper import OUTPUT_CSV_FILE_PROCESSED, NEIGHBORHOOD_SCORES

load_dotenv()
warnings.filterwarnings("ignore")

GEO_APIFY_API_KEY = os.getenv("GEO_APIFY_API_KEY")
LOCALLOGIC_API_TOKEN = os.getenv("LOCALLOGIC_API_TOKEN")

NO_ADDRESS_FOUND = "No address found!"

scraper = cloudscraper.create_scraper()

with open(JSON_FILE_POSTAL_CODE_IDX_MAPPING, "r") as f:
    postal_code_to_idx_dict = json.load(f)

with open(JSON_FILE_POSTAL_CODE_FIRST_3_IDX_MAPPING, "r") as f:
    postal_code_first_3_to_idx_dict = json.load(f)

with open(JSON_FILE_BEST_PARAMS, "r") as f:
    best_params = json.load(f)

input_scaler = joblib.load(PICKLE_FILE_INPUT_SCALER)
rent_scaler = joblib.load(PICKLE_FILE_RENT_SCALER)
model = torch.load(FILE_TORCH_MODEL, map_location=torch.device("cpu"))


def fetch_address_info(text: str):
    """
    Fetch address information from Geoapify API.
    """
    if "prediction" in st.session_state:
        del st.session_state["prediction"]
    st.session_state["address"] = NO_ADDRESS_FOUND

    geoapify_url = f"https://api.geoapify.com/v1/geocode/search?text={text}&apiKey={GEO_APIFY_API_KEY}"

    response = scraper.get(geoapify_url)
    if response.status_code == 200:
        data = response.json()
        if data["features"] and data["features"][0]["properties"]["country"] == "Canada":
            st.session_state["address"] = data["features"][0]["properties"]


def fetch_neighborhood_scores(neighborhood_scores: list[float], longitude: float, latitude: float):
    """
    Fetch neighborhood scores from Locallogic API.
    """
    locallogic_url = f"https://api.locallogic.co/v1/scores?token={LOCALLOGIC_API_TOKEN}&lng={longitude}&lat={latitude}"
    locallogic_response = scraper.get(locallogic_url)

    if locallogic_response.status_code == 200:
        response_scores = locallogic_response.json().get("data").get("attributes")

        if response_scores:
            for attribute in NEIGHBORHOOD_SCORES:
                if attribute in response_scores:
                    neighborhood_scores.append(response_scores[attribute].get("value") / 5)
                else:
                    neighborhood_scores.append(0)


def input():
    st.set_page_config(page_title="Halifax Rental Prediction")
    st.title("Halifax Rental Prediction")

    # Address Input
    address_input = st.text_input("Address")
    postal_code_first_3_idx = 0
    postal_code_idx = 0

    neighborhood_scores = []
    if "address" in st.session_state:
        if st.session_state["address"] == NO_ADDRESS_FOUND:
            st.write(NO_ADDRESS_FOUND)

        else:
            st.write(f'Full address: {st.session_state["address"]["formatted"]}.')

            longitude = st.session_state["address"]["lon"]
            latitude = st.session_state["address"]["lat"]

            # Get postal code from the address and turn it into index
            postal_code = st.session_state["address"]["postcode"]
            if postal_code in postal_code_to_idx_dict:
                postal_code_idx = postal_code_to_idx_dict[postal_code]
            else:
                # If postal code not in the dataset, find the closest postal code
                df = pd.read_csv(OUTPUT_CSV_FILE_PROCESSED)

                min_distance = float("inf")
                for idx, row in df.iterrows():
                    distance = (longitude - row["longitude"]) ** 2 + (latitude - row["latitude"]) ** 2
                    if distance < min_distance:
                        min_distance = distance
                        postal_code = row["postal_code"]
                        postal_code_idx = postal_code_to_idx_dict[postal_code]

            # Get first 3 digits of postal code and turn it into index
            postal_code_first_3 = postal_code[:3]
            postal_code_first_3_idx = postal_code_first_3_to_idx_dict[postal_code_first_3]

            # Get neighborhood scores from Locallogic API
            fetch_neighborhood_scores(neighborhood_scores, longitude, latitude)

    st.button(
        "Check Address",
        on_click=fetch_address_info,
        args=(address_input,),
        type="primary",
    )

    # Bedroom, Bathroom, Area Input
    beds = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=1)
    baths = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=1)
    area = st.number_input("Area (square footage)", min_value=350, max_value=5000, value=1000)

    beds, baths, area = input_scaler.transform([[beds, baths, area]])[0]

    # Additional Features
    additional_feature_dict = {
        "feature_studio": None,
        "feature_pet_friendly": None,
        "feature_furnished": None,
        "feature_fitness_center": None,
        "feature_swimming_pool": None,
        "feature_recreation_room": None,
        "feature_heating": None,
        "feature_water": None,
        "feature_internet": None,
        "feature_ensuite_laundry": None,
        "feature_laundry_room": None,
        "feature_parking": None,
        "feature_underground_parking": None,
    }

    st.write("Additional Features")
    additional_features = []
    for feature in additional_feature_dict:
        if feature in best_params and best_params[feature]:
            feature_name = feature[8:].replace("_", " ").title()
            additional_feature_dict[feature] = st.checkbox(feature_name)
            additional_features.append(1 if additional_feature_dict[feature] else 0)

    input_tensor = None
    if "address" in st.session_state and st.session_state["address"] != NO_ADDRESS_FOUND:
        input_tensor = torch.tensor(
            [
                [
                    postal_code_first_3_idx,
                    postal_code_idx,
                    beds,
                    baths,
                    area,
                    *neighborhood_scores,
                    *additional_features,
                ]
            ]
        ).float()

    st.button(
        "Predict Rent",
        on_click=predict_rent,
        args=(input_tensor,),
        type="primary",
    )

    if "prediction" in st.session_state:
        st.write(st.session_state["prediction"])


def predict_rent(input_tensor: torch.Tensor):
    global model

    if input_tensor != None:
        prediction = model(input_tensor).item()
        prediction = rent_scaler.inverse_transform([[prediction]])[0][0]

        st.session_state["prediction"] = f"Predicted rent: ${prediction:.2f}"
    else:
        st.session_state["prediction"] = "Please check if the address is valid first."


input()
