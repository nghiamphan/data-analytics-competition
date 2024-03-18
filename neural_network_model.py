import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

rent_scaler = MinMaxScaler()


def setup_data():
    input, target = process_data()

    input = torch.tensor(input.values, dtype=torch.float32)
    target = torch.tensor(target.values, dtype=torch.float32)

    # Split the dataset
    input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.2, random_state=42)

    # Create TensorDatasets for the training set
    train_dataset = TensorDataset(input_train, target_train)

    # Create a DataLoader for the training set
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    return dataloader, input_test, target_test


def process_data():
    df = pd.read_csv("data/units_full_info.csv")

    # Process the 'area' column
    process_missing_area(df)

    # Process the "studio" column
    df["studio"] = (df["beds"] == 0).astype(int)

    # Normalize the 'beds', 'baths' and 'area' columns
    df[["beds", "baths", "area"]] = MinMaxScaler().fit_transform(df[["beds", "baths", "area"]])

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

    input = df[
        [
            "beds",
            "baths",
            "area",
            "studio",
            "pet_friendly",
            "furnished",
            "fitness_center",
            "swimming_pool",
            "recreation_room",
            "heating",
            "water",
            "internet",
            "ensuite_laundry",
            "laundry_room",
            "parking",
            "underground_parking",
        ]
    ]

    # Normalize the 'rent' column
    df["rent"] = rent_scaler.fit_transform(df[["rent"]])

    target = df["rent"]
    return input, target


def process_missing_area(df):
    # Calculate the mean 'areas' for each group of 'beds' and 'baths'
    df["mean_area"] = df[(df["area"] != 0) & (df["area"].notna())].groupby(["beds", "baths"])["area"].transform("mean")

    # Extract the unique tuples of 'beds', 'baths' and 'mean_area' if 'area' is not 0 or None
    mean_area_tuples = set(
        df[(df["area"] != 0) & (df["area"].notna())][["beds", "baths", "mean_area"]].itertuples(index=False, name=None)
    )

    # Create a dictionary to store the mean 'area' for each group of 'beds' and 'baths'
    mean_area_dict = {}
    for beds, baths, mean_area in mean_area_tuples:
        mean_area_dict[(beds, baths)] = mean_area

    # Replace the 0 or None 'area' with the mean 'area' for the corresponding group of 'beds' and 'baths'
    for index, row in df.iterrows():
        if row["area"] == 0 or pd.isna(row["area"]):
            if (row["beds"], row["baths"]) in mean_area_dict:
                df.at[index, "area"] = mean_area_dict[(row["beds"], row["baths"])]
            elif not float(row["beds"]).is_integer():
                df.at[index, "area"] = mean_area_dict[(math.floor(row["beds"]), row["baths"])]
            elif not float(row["baths"]).is_integer():
                df.at[index, "area"] = mean_area_dict[(row["beds"], math.floor(row["baths"]))]

    # Drop the 'mean_area' column
    df.drop(columns=["mean_area"], inplace=True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, d_ff=64):
        super(NeuralNetwork, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1),
        )

        self.to(self.device)

    def forward(self, x):
        out = self.feed_forward(x)
        return out


def train_and_test(dataloader, input_test, target_test):
    model = NeuralNetwork(dataloader.dataset.tensors[0].shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(500):
        for input_batch, target_batch in dataloader:

            input_batch = input_batch.to(model.device)
            target_batch = target_batch.to(model.device)

            optimizer.zero_grad()
            outputs = model(input_batch).squeeze()

            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # evaluate
    model.eval()
    with torch.no_grad():
        input_test = input_test.to(model.device)
        target_test = target_test.to(model.device)

        outputs = model(input_test).squeeze()
        loss = criterion(outputs, target_test)
        print(f"\nTest Loss: {loss.item()}")

    predictions = rent_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
    actual = rent_scaler.inverse_transform(target_test.cpu().numpy().reshape(-1, 1)).flatten()

    print("\nPredictions vs Actual")
    for i in range(len(predictions)):
        print(f"{predictions[i]:<10.2f} {actual[i]:.0f}")


def main():
    dataloader, input_test, target_test = setup_data()
    train_and_test(dataloader, input_test, target_test)


if __name__ == "__main__":
    main()
