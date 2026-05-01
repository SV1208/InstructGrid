import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base_path = "logs/film_0429_1025"

def load_scalar(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    tag = ea.Tags()["scalars"][0]
    events = ea.Scalars(tag)
    
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values

# Load data
train_steps, train_loss = load_scalar(os.path.join(base_path, "loss_train"))
val_steps, val_loss     = load_scalar(os.path.join(base_path, "loss_val"))

# ---- Styling for paper ----
plt.figure(figsize=(6, 4))  # compact, paper-friendly

plt.plot(train_steps, train_loss, linewidth=2, label="Training Loss")
plt.plot(val_steps, val_loss, linewidth=2, linestyle="--", label="Validation Loss")

plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Cross-Entropy Loss", fontsize=11)

plt.title("Training and Validation Loss (FiLM Model)", fontsize=12)

plt.legend(frameon=False, fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()

# Save in multiple formats (important for papers)
plt.savefig("film_loss_plot.png", dpi=300)
plt.savefig("film_loss_plot.pdf")   # vector format (BEST for papers)

plt.show()