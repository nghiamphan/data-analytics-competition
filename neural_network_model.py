import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from rentals_ca_scraper import NEIGHBORHOOD_SCORES

CSV_FILE_PROCESSED = "./data/units_info_processed.csv"

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

SEED = 1234

gv_input_scaler = MinMaxScaler()
gv_rent_scaler = MinMaxScaler()

gv_n_postal_codes_first_3 = 0
gv_n_postal_codes = 0


def setup_data():
    input, target = process_data()

    print("Number of samples:", len(input), "\n")

    input = torch.tensor(input.values, dtype=torch.float32)
    target = torch.tensor(target.values, dtype=torch.float32)

    # Split the dataset
    input_train, input_test, target_train, target_test = train_test_split(
        input, target, test_size=0.2, random_state=SEED
    )

    # return dataloader, input_test, target_test
    return input_train, target_train, input_test, target_test


def process_data(raw_csv: str = CSV_FILE_PROCESSED):
    df = pd.read_csv(raw_csv)

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
    df["rent_to_unit_area_ratio"] = df["rent"] / df["area"]
    df = df[(df["rent_to_unit_area_ratio"] > 1.5) & (df["rent_to_unit_area_ratio"] < 4)]

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

    # Normalize the 'rent' column
    df["rent"] = gv_rent_scaler.fit_transform(df[["rent"]])

    # Save the data which will be used for the model to a new CSV file
    df.to_csv("data/units_info_for_model.csv", index=False)

    df_halifax = df[df["city"] == "Halifax"]
    df_halifax.to_csv("data/units_info_for_model_halifax.csv", index=False)

    input = df[INPUT_COLUMNS + NEIGHBORHOOD_SCORES]
    target = df["rent"]
    return input, target


def process_missing_area(df: pd.DataFrame):
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
        input_dim: int,
        n_hidden_layers: int = 1,
        hidden_dim: int = 64,
        postal_code_first_3_dim: int = 4,
        postal_code_dim: int = 2,
        n_postal_codes_first_3: int = 1000,
        n_postal_codes: int = 2000,
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
                nn.Linear(input_dim + postal_code_first_3_dim + postal_code_dim - 2, hidden_dim),
                nn.ReLU(),
            )

            for _ in range(n_hidden_layers - 1):
                self.feed_forward.add_module("hidden", nn.Linear(hidden_dim, hidden_dim))
                self.feed_forward.add_module("relu", nn.ReLU())

            self.feed_forward.add_module("output", nn.Linear(hidden_dim, 1))

        self.to(self.device)

    def forward(self, x: torch.Tensor):
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

    def train_model(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        epochs: int,
        batch_size: int = 8,
        lr: float = 1e-5,
        print_loss: bool = False,
    ):
        dataloader = DataLoader(TensorDataset(input_train, target_train), batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            for input_batch, target_batch in dataloader:

                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                prediction = self(input_batch)

                loss = criterion(prediction, target_batch)
                loss.backward()
                optimizer.step()

            if print_loss and epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def evaluate(self, input_test: torch.Tensor, target_test: torch.Tensor) -> tuple[float, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            input_test = input_test.to(self.device)
            target_test = target_test.to(self.device).unsqueeze(1)

            prediction = self(input_test)
            loss = nn.MSELoss()(prediction, target_test)

        return loss.item(), prediction


def objective(
    trial,
    input_train: torch.Tensor,
    target_train: torch.Tensor,
    epochs: int,
) -> float:

    n_hidden_layers = trial.suggest_int("n_hidden_layers", 0, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512, 1024])
    postal_code_first_3_dim = trial.suggest_categorical("postal_code_first_3_dim", [2, 4, 8, 16])
    postal_code_dim = trial.suggest_categorical("postal_code_dim", [2, 4, 8, 16])

    # Split the training set into a training and validation set
    input_train, input_val, target_train, target_val = train_test_split(
        input_train, target_train, test_size=0.2, random_state=SEED
    )

    model = NeuralNetwork(
        input_train.shape[1],
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        postal_code_first_3_dim=postal_code_first_3_dim,
        postal_code_dim=postal_code_dim,
        n_postal_codes_first_3=gv_n_postal_codes_first_3,
        n_postal_codes=gv_n_postal_codes,
    )

    model.train_model(input_train, target_train, epochs)
    mse = model.evaluate(input_val, target_val)[0]
    return mse


def model_tuning(
    input_train: torch.Tensor,
    target_train: torch.Tensor,
    epochs: int,
    n_trials: int,
) -> dict[str, float]:
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, input_train, target_train, epochs),
        n_trials=n_trials,
    )

    return study.best_params


def print_result(input_test: torch.Tensor, target_test: torch.Tensor, prediction: torch.Tensor):
    # Convert tensors to numpy arrays and reshape
    input_test_np = input_test.cpu().numpy().reshape(-1, input_test.shape[1])

    prediction_np = gv_rent_scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1)).flatten()
    target_test_np = gv_rent_scaler.inverse_transform(target_test.cpu().numpy().reshape(-1, 1)).flatten()

    residual = prediction_np - target_test_np

    df_input_test = pd.DataFrame(input_test_np, columns=INPUT_COLUMNS + NEIGHBORHOOD_SCORES)
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
    input_train, target_train, input_test, target_test = setup_data()

    best_params = model_tuning(input_train, target_train, epochs=100, n_trials=20)

    print("\nBest parameters:", best_params)

    model = NeuralNetwork(
        input_train.shape[1],
        n_postal_codes_first_3=gv_n_postal_codes_first_3,
        n_postal_codes=gv_n_postal_codes,
        **best_params,
    )

    model.train_model(input_train, target_train, epochs=200, print_loss=True)

    test_loss, prediction = model.evaluate(input_test, target_test)
    print("Test Loss:", test_loss)
    print_result(input_test, target_test, prediction)


if __name__ == "__main__":
    main()
