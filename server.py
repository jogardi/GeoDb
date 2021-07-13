from flask import Flask, request, jsonify
import json
from geodb.searchable_kernel import SearchableKernel, Dataset
import torch
from geodb.auto_metric_learn import MetricLearner
from geodb.resources import g_res, from_gcp
from geodb import utils
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = MetricLearner(2, 5, 4)
model.load_state_dict(
    torch.load(
        f"{g_res.models_dir}/mitbih_softmax_62680_feature_extractor.pth",
        map_location=torch.device("cpu"),
    )
)

# ecg_x = utils.standardize(utils.each_with_time(torch.tensor(g_res.mitbih_train_x)))
# ecg_y = g_res.mitbih_train_y
# links = [f"mitbih_train_x.npy#{i}" for i in range(len(ecg_x))]
# ecg_data = Dataset(links, ecg_x, ecg_y)
# sk_ecg = SearchableKernel(model, 5, ecg_data)
caltech_x = torch.tensor(torch.load(from_gcp("ct141embeddings.pt")))
caltech_x -= torch.mean(caltech_x, dim=0)
caltech_x /= torch.std(caltech_x, dim=0)
caltech_links = torch.load(from_gcp("ct141links.pt"))
caltech_labels = torch.load(from_gcp("ct141_labels.pt"))
caltech_data = Dataset(caltech_links, caltech_x, caltech_labels)
sk_caltech = SearchableKernel(lambda x: x, 16, caltech_data)
print("loaded")


@app.route("/search", methods=["POST"])
def search():
    req = request.get_json()
    print("req", req)
    results = sk_caltech.search(
        req["query_link"], radius=req.get("radius", None), k=req.get("k", None)
    )
    print("past req")
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
