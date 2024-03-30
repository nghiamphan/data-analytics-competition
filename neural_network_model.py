import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import joblib
import json

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

    df["postal_code_first_3"] = df["postal_code"].str[:3]
    df["postal_code_first_3_idx"] = df["postal_code_first_3"].astype("category").cat.codes
    df["postal_code_idx"] = df["postal_code"].astype("category").cat.codes

    # Create a dictionary mapping postal codes to indices
    postal_code_idx_mapping = df.set_index("postal_code")["postal_code_idx"].to_dict()
    postal_code_first_3_idx_mapping = df.set_index("postal_code_first_3")["postal_code_first_3_idx"].to_dict()

    # Save the mapping to a JSON file
    with open("saved_model/postal_code_idx_mapping.json", "w") as f:
        json.dump(postal_code_idx_mapping, f, indent=4)
    with open("saved_model/postal_code_first_3_idx_mapping.json", "w") as f:
        json.dump(postal_code_first_3_idx_mapping, f, indent=4)

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

    # Save the scalers
    joblib.dump(gv_input_scaler, "saved_model/input_scaler.pkl")
    joblib.dump(gv_rent_scaler, "saved_model/rent_scaler.pkl")

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
        input_val: torch.Tensor = None,
        target_val: torch.Tensor = None,
        epochs: int = 100,
        min_epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-5,
        l2: float = 1e-5,
        patience: int = 5,
        print_loss: bool = False,
    ):
        """
        Train the model. Model is trained using Adam optimizer and Mean Squared Error loss.
        Early stopping is used to prevent overfitting. If the validation loss does not improve for 'patience' epochs, and the number of epochs is greater than or equal to 'min_epochs', training will stop.

        Parameters
        ----------
        input_train : torch.Tensor
            The input features of the training set.
        target_train : torch.Tensor
            The target values of the training set.
        input_val : torch.Tensor
            The input features of the validation set.
        target_val : torch.Tensor
            The target values of the validation set.
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            The batch size for training.
        lr : float
            The learning rate for the optimizer.
        l2 : float
            The L2 regularization parameter.
        patience : int
            The number of epochs with no improvement before early stopping.
        print_loss : bool
            If True, print the loss every 20 epochs.

        Returns
        -------
        val_loss: float
            The validation loss of the model. Returns None if input_val and target_val are None.
        """
        train_dataloader = DataLoader(TensorDataset(input_train, target_train), batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        best_val_loss = float("inf")
        no_improve_epochs = 0  # Number of epochs with no improvement

        for epoch in range(1, epochs + 1):
            for input_batch, target_batch in train_dataloader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()
                prediction = self(input_batch)

                loss = criterion(prediction, target_batch)
                loss.backward()
                optimizer.step()

            if input_val != None and target_val != None:
                with torch.no_grad():
                    input_val = input_val.to(self.device)
                    target_val = target_val.to(self.device)

                    prediction = self(input_val)
                    val_loss = criterion(prediction, target_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if epoch >= min_epochs and no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if print_loss and epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        if input_val != None and target_val != None:
            return val_loss.item()

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
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024, 2048])
    use_postal_code = trial.suggest_categorical("use_postal_code", [True, False])

    if use_postal_code:
        postal_code_first_3_dim = trial.suggest_categorical("postal_code_first_3_dim", [2, 4, 8, 16])
        postal_code_dim = trial.suggest_categorical("postal_code_dim", [2, 4, 8, 16])
    else:
        postal_code_first_3_dim = 0
        postal_code_dim = 0

    lr = trial.suggest_categorical("lr", [1e-6, 1e-5, 1e-4, 1e-3])
    l2 = trial.suggest_categorical("l2", [0, 1e-6, 1e-5, 1e-4])

    input_train, target_train, input_val, target_val, _, _ = setup_data(df, is_halifax_only=is_halifax_only)

    model = NeuralNetwork(
        input_train.shape[1],
        n_hidden_layers=n_hidden_layers,
        hidden_dim=hidden_dim,
        use_postal_code=use_postal_code,
        postal_code_first_3_dim=postal_code_first_3_dim,
        postal_code_dim=postal_code_dim,
    )

    mse = model.train_model(
        input_train,
        target_train,
        input_val,
        target_val,
        epochs,
        batch_size=512 if is_halifax_only else 16,
        lr=lr,
        l2=l2,
    )
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

    # Save the trials and best parameters
    study.trials_dataframe().to_csv("data/optuna_trials.csv", index=False)
    with open("saved_model/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

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
    print("Mean Error:", residual.mean())
    rmse = (residual**2).mean() ** 0.5
    print("Root Mean Squared Error:", rmse)

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
    plt.xlabel("Actual Value")
    plt.ylabel("Residual")
    plt.title("Difference between Predictions and Actual Values")

    # Show the plot
    plt.grid(True)
    plt.savefig("data/residual_plot.png")
    plt.show()


def main(tune_model: bool = True):
    df = process_data()

    if tune_model:
        best_params = model_tuning(df, epochs=100, n_trials=100)
    else:
        with open("saved_model/best_params.json", "r") as f:
            best_params = json.load(f)

    print("\nBest parameters:", best_params)

    additional_columns = []
    for column in ADDITIONAL_COLUMNS:
        if best_params[f"feature_{column}"]:
            additional_columns.append(column)

    input_train, target_train, input_val, target_val, input_test, target_test = setup_data(
        df,
        is_halifax_only=best_params["is_halifax_only"],
        input_columns=ESSENTIAL_COLUMNS + additional_columns,
    )

    model = NeuralNetwork(
        input_train.shape[1],
        n_hidden_layers=best_params["n_hidden_layers"],
        hidden_dim=best_params["hidden_dim"],
        use_postal_code=best_params["use_postal_code"],
        postal_code_first_3_dim=best_params["postal_code_first_3_dim"] if best_params["use_postal_code"] else 0,
        postal_code_dim=best_params["postal_code_dim"] if best_params["use_postal_code"] else 0,
    )

    model.train_model(
        input_train,
        target_train,
        input_val,
        target_val,
        epochs=200,
        batch_size=512 if best_params["is_halifax_only"] else 16,
        lr=best_params["lr"],
        l2=best_params["l2"],
        print_loss=True,
    )

    # Save the model
    torch.save(model, "saved_model/nn_model.pt")

    test_loss, prediction = model.evaluate(input_test, target_test)
    print("Test Loss:", test_loss)
    print_result(input_test, target_test, prediction, additional_columns)


if __name__ == "__main__":
    main(tune_model=True)
