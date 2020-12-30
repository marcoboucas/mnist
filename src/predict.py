"""Load the model and make the prediction."""
import os
import pandas as pd
import torch
from tqdm import tqdm

BATCH_SIZE = 2


def predict():
    """Predict."""
    # GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = torch.load('model.pt')
    model.to(device)
    model.eval()

    # Load the data and predict
    dataset = pd.read_csv(os.path.join('data', "test.csv")).to_numpy()
    dataset = torch.tensor(
        (dataset.astype("float")/255).reshape((-1, 1,  28, 28)), dtype=torch.float32
    )

    model_results = []
    for i, data in enumerate(tqdm(torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)), 0):
        output = model(data.to(device))
        output = output.argmax(dim=1).tolist()
        model_results.extend(output)

    pd.DataFrame(
        {
            "ImageId": [i+1 for i in range(len(model_results))],
            "Label": model_results
        }
    ).to_csv('model_results.csv', index=False)


if __name__ == "__main__":
    predict()
