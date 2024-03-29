import streamlit as st
import cloudscraper
import torch
import joblib
import json
import os
import warnings

from dotenv import load_dotenv
from neural_network_model import NeuralNetwork, NEIGHBORHOOD_SCORES

load_dotenv()
warnings.filterwarnings("ignore")

GEO_APIFY_API_KEY = os.getenv("GEO_APIFY_API_KEY")
LOCALLOGIC_API_TOKEN = os.getenv("LOCALLOGIC_API_TOKEN")

NO_ADDRESS_FOUND = "No address found!"

scraper = cloudscraper.create_scraper()

with open("saved_model/postal_code_idx_mapping.json", "r") as f:
    postal_code_to_idx_dict = json.load(f)

with open("saved_model/postal_code_first_3_idx_mapping.json", "r") as f:
    postal_code_first_3_to_idx_dict = json.load(f)

with open("saved_model/best_params.json", "r") as f:
    best_params = json.load(f)

input_scaler = joblib.load("saved_model/input_scaler.pkl")
rent_scaler = joblib.load("saved_model/rent_scaler.pkl")
model = joblib.load("saved_model/nn_model.pkl")


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
    # st.set_page_config(page_title="Halifax Rental Prediction")
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
            text_to_display = f'Address: {st.session_state["address"]["formatted"]}.'

            # Get postal code from the address and turn it into index
            postal_code = st.session_state["address"]["postcode"]
            if postal_code in postal_code_to_idx_dict:
                postal_code_idx = postal_code_to_idx_dict[postal_code]
            else:

                text_to_display = (
                    f"{text_to_display} Warning: Postal code not exist in the model. Prediction may not be accurate."
                )

            # Get first 3 digits of postal code and turn it into index
            postal_code_first_3 = postal_code[:3]
            if postal_code_first_3 in postal_code_first_3_to_idx_dict:
                postal_code_first_3_idx = postal_code_first_3_to_idx_dict[postal_code_first_3]
            else:
                postal_code_first_3_idx = 0

            st.write(text_to_display)

            # Get neighborhood scores from Locallogic API
            longitude = st.session_state["address"]["lon"]
            latitude = st.session_state["address"]["lat"]
            fetch_neighborhood_scores(neighborhood_scores, longitude, latitude)

    st.button(
        "Check Address",
        on_click=fetch_address_info,
        args=(address_input,),
    )

    # Bedroom, Bathroom, Area Input
    beds = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=1)
    baths = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=1)
    area = st.number_input("Area (square footage)", min_value=350, max_value=5000, value=1000)

    beds, baths, area = input_scaler.transform([[beds, baths, area]])[0]

    # Additional Features
    st.write("Additional Features")
    studio = st.checkbox("Is Studio")
    pet_friendly = st.checkbox("Pets Friendly")
    furnished = st.checkbox("Furnished")
    fitness_center = st.checkbox("Fitness Center")
    swimming_pool = st.checkbox("Swimming Pool")
    recreation_room = st.checkbox("Recreation Room")
    heating = st.checkbox("Heating Included")
    water = st.checkbox("Water Included")
    internet = st.checkbox("Internet Included")
    ensuite_laundry = st.checkbox("Ensuite Laundry")
    laundry_room = st.checkbox("Laundry Room")
    parking = st.checkbox("Parking")
    underground_parking = st.checkbox("Underground Parking")

    additional_feature_dict = {
        "feature_studio": studio,
        "feature_pet_friendly": pet_friendly,
        "feature_furnished": furnished,
        "feature_fitness_center": fitness_center,
        "feature_swimming_pool": swimming_pool,
        "feature_recreation_room": recreation_room,
        "feature_heating": heating,
        "feature_water": water,
        "feature_internet": internet,
        "feature_ensuite_laundry": ensuite_laundry,
        "feature_laundry_room": laundry_room,
        "feature_parking": parking,
        "feature_underground_parking": underground_parking,
    }

    additional_features = []
    for feature in additional_feature_dict:
        if best_params[feature]:
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
    )

    if "prediction" in st.session_state:
        st.write(st.session_state["prediction"])


def predict_rent(input_tensor: torch.Tensor):
    global model

    if input_tensor != None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_tensor = input_tensor.to(device)

        rent_scaler = joblib.load("saved_model/rent_scaler.pkl")

        prediction = model(input_tensor).item()
        prediction = rent_scaler.inverse_transform([[prediction]])[0][0]

        st.session_state["prediction"] = f"Predicted rent: ${prediction:.2f}"
    else:
        st.session_state["prediction"] = "Please check if the address is valid first."


input()
