import matplotlib.pyplot as plt
import numpy as np

epochs = [1, 2, 3, 4]

# Qwen1.5-1.8B-Chat runs
qwen15_chat_1 = [0.197457533, 0.233785961, 0.233795423, 0.244980034]
qwen15_chat_2 = [None, 0.243125143, 0.248644876, 0.25393567]

# Qwen2.5-7B-Instruct runs
qwen25_inst_1 = [0.186845474, 0.204282666, 0.212277764, 0.217275831]
qwen25_inst_2 = [None, 0.199651216, 0.202781854, 0.209874838]

def compute_stats(run1, run2):
    means, stds = [], []
    for v1, v2 in zip(run1, run2):
        if v2 is None:  # only one run
            vals = [v1]
        else:
            vals = [v1, v2]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

# Compute mean and std for both models
qwen15_mean, qwen15_std = compute_stats(qwen15_chat_1, qwen15_chat_2)
qwen25_mean, qwen25_std = compute_stats(qwen25_inst_1, qwen25_inst_2)
# Plot
plt.figure(figsize=(8, 6))

# Qwen1.5-1.8B-Chat
plt.plot(epochs, qwen15_mean, marker='o', color='blue', label='Qwen1.5-1.8B-Chat')
plt.fill_between(epochs,
                 qwen15_mean - qwen15_std,
                 qwen15_mean + qwen15_std,
                 color='blue', alpha=0.2)

# Qwen2.5-7B-Instruct
plt.plot(epochs, qwen25_mean, marker='s', color='green', label='Qwen2.5-7B-Instruct')
plt.fill_between(epochs,
                 qwen25_mean - qwen25_std,
                 qwen25_mean + qwen25_std,
                 color='green', alpha=0.2)

# Labels and title

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel("Epochs", fontsize=18)
plt.ylabel("Resemblance Score", fontsize=18)
plt.title("Resemblance vs. Epochs", fontsize=20)
plt.legend(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("resemblance_vs_epochs.pdf")
plt.show()
