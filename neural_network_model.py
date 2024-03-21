import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

INPUT_COLUMNS = [
    "postal_code_first_3_idx",
    "postal_code_idx",
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

gv_input_scaler = MinMaxScaler(feature_range=(1, 3))
gv_rent_scaler = MinMaxScaler()

gv_n_postal_codes_first_3 = 0
gv_n_postal_codes = 0


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

    # df = df[df["city"] == "Ottawa"]

    # Filter apartments
    df = df[df["property_type"] == "apartment"]

    # Process the 'area' column
    df.loc[df["area"] < 300, "area"] = 0

    process_missing_area(df)
    df = df[(df["area"].notna()) & (df["area"] != 0) & (df["rent"].notna()) & (df["rent"] != 0)]

    # Process the "studio" column
    df["studio"] = (df["beds"] == 0).astype(int)
    df.loc[df["beds"] == 0, "beds"] = 1

    # Remove outliers based on the rent-to-area ratio
    df["rent_to_area"] = df["rent"] / df["area"]
    df = df[(df["rent_to_area"] > 1) & (df["rent_to_area"] < 4)]

    # Convert postal_code to category and then to its corresponding codes
    df = df[df["postal_code"].notna()]
    df["postal_code_first_3_idx"] = df["postal_code"].astype(str).str[:3].astype("category").cat.codes
    df["postal_code_idx"] = df["postal_code"].astype("category").cat.codes

    global gv_n_postal_codes_first_3
    global gv_n_postal_codes
    gv_n_postal_codes_first_3 = df["postal_code_first_3_idx"].nunique()
    gv_n_postal_codes = df["postal_code_idx"].nunique()

    # Normalize the 'beds', 'baths' and 'area' columns
    df[["beds", "baths", "area"]] = gv_input_scaler.fit_transform(df[["beds", "baths", "area"]])

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

    print("Dataset size:", len(df))

    input = df[INPUT_COLUMNS]

    # Normalize the 'rent' column
    df["rent"] = gv_rent_scaler.fit_transform(df[["rent"]])

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

    # Drop the 'mean_area' column
    df.drop(columns=["mean_area"], inplace=True)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        n_hidden_layers=1,
        d_ff=64,
        postal_code_first_3_dim=4,
        postal_code_dim=2,
        n_postal_codes_first_3=1000,
        n_postal_codes=2000,
    ):
        super(NeuralNetwork, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.postal_code_embedding_first_3 = nn.Embedding(n_postal_codes_first_3, postal_code_first_3_dim)
        self.postal_code_embedding = nn.Embedding(n_postal_codes, postal_code_dim)

        if n_hidden_layers == 0:
            self.feed_forward = nn.Linear(input_dim + postal_code_first_3_dim + postal_code_dim - 2, 1)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(input_dim + postal_code_first_3_dim + postal_code_dim - 2, d_ff),
                nn.ReLU(),
            )

            for _ in range(n_hidden_layers - 1):
                self.feed_forward.add_module("hidden", nn.Linear(d_ff, d_ff))
                self.feed_forward.add_module("relu", nn.ReLU())

            self.feed_forward.add_module("output", nn.Linear(d_ff, 1))

        self.to(self.device)

    def forward(self, x):
        # Split the input into the postal code and other features
        postal_code_first_3 = x[:, 0].long()
        postal_code = x[:, 1].long()
        other_features = x[:, 2:]

        # Embed the postal codes
        postal_code_first_3 = self.postal_code_embedding_first_3(postal_code_first_3)
        postal_code = self.postal_code_embedding(postal_code)

        # Concatenate the embeddings with the other features
        x = torch.cat((postal_code_first_3, postal_code, other_features), dim=1)

        out = self.feed_forward(x)
        return out


def train_and_test(dataloader, input_test, target_test):
    model = NeuralNetwork(
        dataloader.dataset.tensors[0].shape[1],
        n_postal_codes_first_3=gv_n_postal_codes_first_3,
        n_postal_codes=gv_n_postal_codes,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(200):
        for input_batch, target_batch in dataloader:

            input_batch = input_batch.to(model.device)
            target_batch = target_batch.to(model.device).unsqueeze(1)

            optimizer.zero_grad()
            prediction = model(input_batch)

            loss = criterion(prediction, target_batch)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # evaluate
    model.eval()
    with torch.no_grad():
        input_test = input_test.to(model.device)
        target_test = target_test.to(model.device).unsqueeze(1)

        prediction = model(input_test)
        loss = criterion(prediction, target_test)
        print(f"\nTest Loss: {loss.item()}")
        print_result(input_test, target_test, prediction)


def print_result(input_test: torch.Tensor, target_test: torch.Tensor, prediction: torch.Tensor):
    # Convert tensors to numpy arrays and reshape
    input_test_np = input_test.cpu().numpy().reshape(-1, input_test.shape[1])

    prediction_np = gv_rent_scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1)).flatten()
    target_test_np = gv_rent_scaler.inverse_transform(target_test.cpu().numpy().reshape(-1, 1)).flatten()

    residual = prediction_np - target_test_np

    df_input_test = pd.DataFrame(input_test_np, columns=INPUT_COLUMNS)
    df_input_test[["beds", "baths", "area"]] = gv_input_scaler.inverse_transform(
        df_input_test[["beds", "baths", "area"]]
    )

    df_target_test = pd.DataFrame(target_test_np, columns=["actual rent"])
    df_prediction = pd.DataFrame(prediction_np, columns=["prediction"])
    df_residual = pd.DataFrame(residual, columns=["residual"])

    df = pd.concat([df_input_test, df_target_test, df_prediction, df_residual], axis=1)

    df = df.sort_values(by="residual")

    df.to_csv("data/predictions.csv", index=False)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(target_test_np, residual, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title("Difference between Predictions and Actual Values")

    # Show the plot
    plt.show()


def main():
    dataloader, input_test, target_test = setup_data()
    train_and_test(dataloader, input_test, target_test)


if __name__ == "__main__":
    main()
