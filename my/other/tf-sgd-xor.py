import time
import math
import numpy as np

layers=[
    {"weights": np.random.random((2, 2)), "biases": np.random.random((2))},
    {"weights": np.random.random((2, 2)), "biases": np.random.random((2))},
    {"weights": np.random.random((2, 1)), "biases": np.random.random((1))},
]

trains=[
    {"inputs": (0, 0), "target": (0 ^ 0)},
    {"inputs": (0, 1), "target": (0 ^ 1)},
    {"inputs": (1, 0), "target": (1 ^ 0)},
    {"inputs": (1, 1), "target": (1 ^ 1)}
]

mse = lambda y, t: ((y - t) ** 2).mean()
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: x * (1 - x)

def forward(inp):
    for layer in layers:
        layer["inputs"] = inp
        layer["outputs"] = layer["biases"]
        for i in range(len(layer["weights"])):
            layer["outputs"] = layer["outputs"] + layer["inputs"][i] * layer["weights"][i]
        layer["outputs"] = sigmoid(layer["outputs"])
        inp = layer["outputs"]
    return layers[-1]["outputs"]

def backprop(cfg):
    for e in range(cfg["epochs"]):
        loss = 0
        for train in trains:
            out = forward(train["inputs"])
            loss = loss + mse(out, train["target"])
            grads = out - train["target"]
            for layer in layers[::-1]:
                net = grads * dsigmoid(layer["outputs"])
                grads = [sum(weights * net) for weights in layer["weights"]]
                layer["biases"] = layer["biases"] - cfg["step"] * net
                layer["weights"] = [weights - cfg["step"] * net * layer["inputs"] for weights in layer["weights"]]
        loss = loss / len(trains)
        print(f"{e}# loss = {loss}")
        if loss < cfg["error"]:
            break
    return (e, loss)

time0 = time.time()  # Record the start time

stats = backprop({
    "epochs": 10000,
    "error": 0.01,
    "step": 0.1
})

for train in trains:
    predict = forward(train["inputs"])
    print(np.round(predict), train["target"])

print(f"Execution time: {time.time() - time0} seconds. {round(stats[0] / (time.time() - time0))} epochs per second")
