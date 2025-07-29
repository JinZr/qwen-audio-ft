import pickle
import os
import numpy as np
from sklearn import metrics

result_dir = "results/sr-level/02/qwen-audio/lr3e-5"

result_output = pickle.load(
    open(os.path.join(result_dir, "results.pkl"), "rb"), encoding="utf-8"
)
test_result = result_output["test_results"]
label_dic = np.load(
    "/mnt/nvme_share/cuizy/SA-detection/label/sr-level.npy", allow_pickle=True
).item()

pred_dic = {}
for sample in test_result:
    id_temp = sample["id"].split("-")
    id = "{}-{}-{}".format(id_temp[0], id_temp[1], id_temp[2])
    if id in pred_dic:
        pred_dic[id].append(sample["hard_label"])
    else:
        pred_dic[id] = [sample["hard_label"]]

y_true = []
y_pred = []
for key in pred_dic:
    y_true.append(label_dic[key])
    y_pred.append(round(np.mean(pred_dic[key])))

print(metrics.classification_report(y_true, y_pred))
print(metrics.confusion_matrix(y_true, y_pred))

with open(os.path.join(result_dir, "confusion_matrix.log"), "w") as f:
    f.write(str(metrics.classification_report(y_true, y_pred)))
    f.write("\n")
    f.write(str(metrics.confusion_matrix(y_true, y_pred)))
