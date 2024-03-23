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

# The file contains the data that was lightly processed. It is the final output file from rentals_ca_scraper.py
CSV_FILE_PROCESSED = "./data/units_info_processed.csv"

ESSENTIAL_COLUMNS = [
    "postal_code_first_3_idx",
    "postal_code_idx",
    "beds",
    "baths",
    "area",
] + NEIGHBORHOOD_SCORES

ADDITIONAL_COLUMNS = [
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

HALIFAX = ["Halifax", "Bedford", "Dartmouth"]

SEED = 1234

gv_input_scaler = MinMaxScaler()
gv_rent_scaler = MinMaxScaler()

gv_n_postal_codes_first_3 = 0
gv_n_postal_codes = 0


def setup_data(
    df: pd.DataFrame,
    is_halifax_only: bool = False,
    input_columns: list[str] = ESSENTIAL_COLUMNS + ADDITIONAL_COLUMNS,
) -> tuple:
    """
    Split the data into training, validation and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    is_halifax_only : bool
        If True, only use the data from Halifax, Bedford and Dartmouth. Default: False.
    input_columns : list[str]
        The columns to use as input features.

    Returns
    -------
    input_train : torch.Tensor
    target_train : torch.Tensor
    input_val : torch.Tensor
    target_val : torch.Tensor
    input_test : torch.Tensor
    target_test : torch.Tensor
    """
    # Split the data into training and test sets
    df_halifax = df[df["city"].isin(HALIFAX)]

    if is_halifax_only:
        # train 60%, validation 10%, test 30%
        df_train, df_temp = train_test_split(df_halifax, test_size=0.4, random_state=SEED)
        df_val, df_test = train_test_split(df_temp, test_size=0.75, random_state=SEED)

        assert len(df_train) + len(df_val) + len(df_test) == len(df_halifax)
    else:
        # train: 60% halifax + other cities, validation: 10% halifax, test: 30% halifax
        _, df_temp = train_test_split(df_halifax, test_size=0.4, random_state=SEED)
        df_val, df_test = train_test_split(df_temp, test_size=0.75, random_state=SEED)
        df_train = df[~df.index.isin(df_val.index.union(df_test.index))]

        assert len(df_train) + len(df_val) + len(df_test) == len(df)

    input_train = df_train[input_columns]
    target_train = df_train["rent"]
    input_val = df_val[input_columns]
    target_val = df_val["rent"]
    input_test = df_test[input_columns]
    target_test = df_test["rent"]

    print("\nNumber of samples in the training set:", len(input_train))
    print("Number of samples in the validation set:", len(input_val))
    print("Number of samples in the test set:", len(input_test))

    # Convert the data to PyTorch tensors
    input_train = torch.tensor(input_train.values, dtype=torch.float32)
    target_train = torch.tensor(target_train.values, dtype=torch.float32).unsqueeze(1)
    input_val = torch.tensor(input_val.values, dtype=torch.float32)
    target_val = torch.tensor(target_val.values, dtype=torch.float32).unsqueeze(1)
    input_test = torch.tensor(input_test.values, dtype=torch.float32)
    target_test = torch.tensor(target_test.values, dtype=torch.float32).unsqueeze(1)

    return input_train, target_train, input_val, target_val, input_test, target_test


def process_data(csv: str = CSV_FILE_PROCESSED) -> pd.DataFrame:
    """
    Process the data before feeding them into the model.

    Parameters
    ----------
    csv : str
        The path to the CSV file. This file should be the final output file from rentals_ca_scraper.py.

    Returns
    -------
    df : pd.DataFrame
        The processed DataFrame.
    """
    df = pd.read_csv(csv)

    # Filter apartments
    df = df[df["property_type"] == "apartment"]

    # Filter out units with 0 baths
    df = df[df["baths"] > 0]

    # Process the 'area' column
    df = df[~((df["area"] > 0) & (df["area"] < 350))]

    process_missing_area(df)
    df = df[(df["area"].notna()) & (df["area"] != 0) & (df["rent"].notna()) & (df["rent"] != 0)]

    # Process the "studio" column
    df["studio"] = (df["beds"] == 0).astype(int)
    df.loc[df["beds"] == 0, "beds"] = 1

    # Remove outliers based on the rent-to-area ratio
    df["rent_to_unit_area_ratio"] = df["rent"] / df["area"]
    df = df[(df["rent_to_unit_area_ratio"] > 1.5) & (df["rent_to_unit_area_ratio"] < 6)]

    # Convert postal_code to category and then to its corresponding codes
    df = df[df["postal_code"].notna()]
    df["postal_code_first_3_idx"] = df["postal_code"].astype(str).str[:3].astype("category").cat.codes
    df["postal_code_idx"] = df["postal_code"].astype("category").cat.codes

    global gv_n_postal_codes_first_3
    global gv_n_postal_codes
    gv_n_postal_codes_first_3 = df["postal_code_first_3_idx"].nunique()
    gv_n_postal_codes = df["postal_code_idx"].nunique()

    # Save the data which will be used for the model to a new CSV file
    df.to_csv("data/units_info_for_model.csv", index=False)

    df_halifax = df[df["city"].isin(HALIFAX)]
    df_halifax.to_csv("data/units_info_for_model_halifax.csv", index=False)

    # Normalize the 'beds', 'baths' and 'area' columns
    df[["beds", "baths", "area"]] = gv_input_scaler.fit_transform(df[["beds", "baths", "area"]])

    # Normalize the 'rent' column
    df["rent"] = gv_rent_scaler.fit_transform(df[["rent"]])

    return df


def process_missing_area(df: pd.DataFrame):
    """
    Estimate the missing values in the 'area' column. Change the values in-place for the DataFrame parameter.
    """
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
        hidden_dim: int = 256,
        use_postal_code: bool = False,
        postal_code_first_3_dim: int = 4,
        postal_code_dim: int = 2,
        n_postal_codes_first_3: int = 1000,
        n_postal_codes: int = 2000,
    ):
        """
        Parameters
        ----------
        input_dim : int
            The number of input features.
        n_hidden_layers : int
            The number of hidden layers. Default: 1.
        hidden_dim : int
            The number of neurons in each hidden layer. Default: 256.
        use_postal_code : bool
            If True, use the postal code columns as an input feature. Default: False.
        postal_code_first_3_dim : int
            The dimension of the postal code embedding for the first 3 characters. Default: 4.
        postal_code_dim : int
            The dimension of the postal code embedding. Default: 2.
        n_postal_codes_first_3 : int
            The number of unique postal codes for the first 3 characters. Default: 1000.
        n_postal_codes : int
            The number of unique postal codes. Default: 2000.
        """
        super(NeuralNetwork, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_postal_code = use_postal_code
        if use_postal_code:
            global gv_n_postal_codes_first_3
            global gv_n_postal_codes

            if gv_n_postal_codes_first_3 > 0:
                n_postal_codes_first_3 = gv_n_postal_codes_first_3
            if gv_n_postal_codes > 0:
                n_postal_codes = gv_n_postal_codes

            self.postal_code_embedding_first_3 = nn.Embedding(n_postal_codes_first_3, postal_code_first_3_dim)
            self.postal_code_embedding = nn.Embedding(n_postal_codes, postal_code_dim)
            input_dim += postal_code_first_3_dim + postal_code_dim - 2
        else:
            input_dim -= 2

        if n_hidden_layers == 0:
            self.feed_forward = nn.Linear(input_dim, 1)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )

            for _ in range(n_hidden_layers - 1):
                self.feed_forward.add_module("hidden", nn.Linear(hidden_dim, hidden_dim))
                self.feed_forward.add_module("relu", nn.ReLU())

            self.feed_forward.add_module("output", nn.Linear(hidden_dim, 1))

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        if self.use_postal_code:
            # Split the input into the postal code and other features
            postal_code_first_3 = x[:, 0].long()
            postal_code = x[:, 1].long()
            other_features = x[:, 2:]

            # Embed the postal codes
            postal_code_first_3 = self.postal_code_embedding_first_3(postal_code_first_3)
            postal_code = self.postal_code_embedding(postal_code)

            # Concatenate the embeddings with the other features
            x = torch.cat((postal_code_first_3, postal_code, other_features), dim=1)
        else:
            x = x[:, 2:]

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
                target_batch = target_batch.to(self.device)

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
            target_test = target_test.to(self.device)

            prediction = self(input_test)
            loss = nn.MSELoss()(prediction, target_test)

        return loss.item(), prediction


def objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    epochs: int,
) -> float:
    """
    Objective function for optuna.

    Parameters
    ----------
    trial : optuna.Trial
        The trial object.
    df : pd.DataFrame
        The DataFrame containing the data.
    epochs : int
        The number of epochs to train the model.

    Returns
    -------
    mse : float
        The mean squared error of the model on the validation set.
    """
    is_halifax_only = trial.suggest_categorical("is_halifax_only", [True, False])

    chosen_features = []
    for i in range(len(ADDITIONAL_COLUMNS)):
        if trial.suggest_categorical(f"feature_{ADDITIONAL_COLUMNS[i]}", [True, False]):
            chosen_features.append(ADDITIONAL_COLUMNS[i])

    n_hidden_layers = trial.suggest_int("n_hidden_layers", 0, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512, 1024])
    use_postal_code = trial.suggest_categorical("use_postal_code", [True, False])
    postal_code_first_3_dim = trial.suggest_categorical("postal_code_first_3_dim", [2, 4, 8, 16])
    postal_code_dim = trial.suggest_categorical("postal_code_dim", [2, 4, 8, 16])

    input_train, target_train, input_val, target_val, _, _ = setup_data(df, is_halifax_only=is_halifax_only)

    model = NeuralNetwork(
        input_train.shape[1],
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        use_postal_code=use_postal_code,
        postal_code_first_3_dim=postal_code_first_3_dim,
        postal_code_dim=postal_code_dim,
    )

    model.train_model(input_train, target_train, epochs)
    mse = model.evaluate(input_val, target_val)[0]
    return mse


def model_tuning(
    df: pd.DataFrame,
    epochs: int,
    n_trials: int,
) -> dict[str, float]:
    """
    Tune the data selection and hyperparameters of the model.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    epochs : int
        The number of epochs to train the model in each trial.
    n_trials : int
        The number of experiments to run.

    Returns
    -------
    best_params : dict[str, float]
        The best parameters found by optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, df, epochs),
        n_trials=n_trials,
    )

    return study.best_params


def print_result(
    input_test: torch.Tensor,
    target_test: torch.Tensor,
    prediction: torch.Tensor,
    addtional_columns: list = ADDITIONAL_COLUMNS,
):
    # Convert tensors to numpy arrays and reshape
    input_test_np = input_test.cpu().numpy().reshape(-1, input_test.shape[1])

    prediction_np = gv_rent_scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1)).flatten()
    target_test_np = gv_rent_scaler.inverse_transform(target_test.cpu().numpy().reshape(-1, 1)).flatten()

    residual = prediction_np - target_test_np

    df_input_test = pd.DataFrame(input_test_np, columns=ESSENTIAL_COLUMNS + addtional_columns)
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
    df = process_data()

    best_params = model_tuning(df, epochs=100, n_trials=1)

    print("\nBest parameters:", best_params)

    additional_columns = []
    for column in ADDITIONAL_COLUMNS:
        if best_params[f"feature_{column}"]:
            additional_columns.append(column)

    input_train, target_train, _, _, input_test, target_test = setup_data(
        df,
        is_halifax_only=best_params["is_halifax_only"],
        input_columns=ESSENTIAL_COLUMNS + additional_columns,
    )

    model = NeuralNetwork(
        input_train.shape[1],
        n_hidden_layers=best_params["n_hidden_layers"],
        hidden_dim=best_params["hidden_dim"],
        use_postal_code=best_params["use_postal_code"],
        postal_code_first_3_dim=best_params["postal_code_first_3_dim"],
        postal_code_dim=best_params["postal_code_dim"],
    )

    model.train_model(input_train, target_train, epochs=200, print_loss=True)

    test_loss, prediction = model.evaluate(input_test, target_test)
    print("Test Loss:", test_loss)
    print_result(input_test, target_test, prediction, additional_columns)


if __name__ == "__main__":
    main()
