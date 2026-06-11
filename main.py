from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


def sanitize_text(text: str) -> str:
    replacements = {
        str(Path.cwd()): ".",
        str(Path.home()): "~",
    }
    sanitized = text
    for original, replacement in replacements.items():
        sanitized = sanitized.replace(original, replacement)
    return sanitized


def show_clean_exception(
    exception_type: type[BaseException],
    exception: BaseException,
    traceback: object,
) -> None:
    if "--debug" in sys.argv:
        sys.__excepthook__(exception_type, exception, traceback)
        return
    if issubclass(exception_type, KeyboardInterrupt):
        print("\nStopped by user.", file=sys.stderr)
        return
    print(f"\nError: {sanitize_text(str(exception))}", file=sys.stderr)
    print("Run again with --debug to show the full traceback.", file=sys.stderr)


def show_clean_warning(
    message: Warning,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: object | None = None,
    line: str | None = None,
) -> None:
    output = file if file is not None else sys.stderr
    warning_text = sanitize_text(str(message))
    print(f"Warning: {category.__name__}: {warning_text}", file=output)


sys.excepthook = show_clean_exception
if "--debug" not in sys.argv:
    warnings.showwarning = show_clean_warning

os.environ.setdefault("MPLCONFIGDIR", "/tmp/entropy-confidence-lab-matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

EPSILON = 1e-12
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


def configure_warning_output(debug: bool) -> None:
    if debug:
        return

    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    warnings.showwarning = show_clean_warning


class ProgressBar:
    def __init__(self, label: str, total_steps: int, enabled: bool = True) -> None:
        self.label = label
        self.total_steps = max(total_steps, 1)
        self.enabled = enabled and sys.stderr.isatty()
        self.started_at = time.monotonic()
        self.width = 28
        self.last_length = 0

    def update(
        self,
        step: int,
        loss: float | None = None,
        accuracy: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        progress = min(max(step / self.total_steps, 0.0), 1.0)
        filled_width = round(progress * self.width)
        bar = "█" * filled_width + "░" * (self.width - filled_width)
        elapsed_seconds = time.monotonic() - self.started_at
        fields = [
            f"\r{self.label}",
            f"[{bar}]",
            f"{step:>3}/{self.total_steps:<3}",
            f"{progress * 100:5.1f}%",
            f"{elapsed_seconds:5.1f}s",
        ]
        if loss is not None:
            fields.append(f"loss {loss:.4f}")
        if accuracy is not None:
            fields.append(f"acc {accuracy:.4f}")

        line = " | ".join(fields)
        padding = " " * max(self.last_length - len(line), 0)
        print(line + padding, end="", file=sys.stderr, flush=True)
        self.last_length = len(line)

    def close(self) -> None:
        if self.enabled:
            print(file=sys.stderr, flush=True)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class PredictionBundle:
    images: torch.Tensor
    labels: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor
    confidences: torch.Tensor
    entropies: torch.Tensor
    cross_entropies: torch.Tensor
    correct: torch.Tensor


class FashionMNISTMLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class FashionMNISTCNN(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "mlp":
        return FashionMNISTMLP(args.hidden_size, args.dropout)
    return FashionMNISTCNN(args.dropout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Fashion-MNIST classifier and analyze entropy/cross-entropy."
    )
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_argument: str) -> torch.device:
    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_argument == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device, device_argument: str) -> str:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        return f"Using device: cuda ({device_name})"
    if device_argument == "auto":
        return "Using device: cpu (CUDA not available, using CPU fallback)"
    return "Using device: cpu"


def stable_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    log_normalizer = torch.logsumexp(logits, dim=1, keepdim=True)
    return logits - log_normalizer


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted_logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)


def manual_cross_entropy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    log_probabilities = stable_log_softmax(logits)
    row_indices = torch.arange(labels.size(0), device=labels.device)
    true_class_log_probabilities = log_probabilities[row_indices, labels]
    return -true_class_log_probabilities.mean()


def entropy_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    safe_probabilities = probabilities.clamp_min(EPSILON)
    return -(probabilities * torch.log(safe_probabilities)).sum(dim=1)


def cross_entropy_from_probabilities(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    row_indices = torch.arange(labels.size(0), device=labels.device)
    true_class_probabilities = probabilities[row_indices, labels].clamp_min(EPSILON)
    return -torch.log(true_class_probabilities)


def build_data_loaders(
    args: argparse.Namespace, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    train_transform_steps = []
    if args.augment:
        train_transform_steps.extend(
            [
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]
    )
    train_transform = transforms.Compose(train_transform_steps)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=test_transform,
    )

    if args.train_limit is not None:
        train_dataset = Subset(train_dataset, range(args.train_limit))
    if args.test_limit is not None:
        test_dataset = Subset(test_dataset, range(args.test_limit))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = ProgressBar(
        f"Epoch {epoch:02d}/{total_epochs} train", len(data_loader), show_progress
    )

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = manual_cross_entropy_from_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size
        progress.update(
            batch_index,
            loss=total_loss / total_count,
            accuracy=total_correct / total_count,
        )

    progress.close()

    return EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label: str = "Evaluating",
    show_progress: bool = False,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    progress = ProgressBar(label, len(data_loader), show_progress)

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = manual_cross_entropy_from_logits(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size
        progress.update(
            batch_index,
            loss=total_loss / total_count,
            accuracy=total_correct / total_count,
        )

    progress.close()

    return EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    show_progress: bool,
) -> PredictionBundle:
    model.eval()
    images_list = []
    labels_list = []
    predictions_list = []
    probabilities_list = []
    confidences_list = []
    entropies_list = []
    cross_entropies_list = []
    correct_list = []
    progress = ProgressBar("Analyzing test predictions", len(data_loader), show_progress)

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probabilities = stable_softmax(logits)
        confidences, predictions = probabilities.max(dim=1)
        entropies = entropy_from_probabilities(probabilities)
        cross_entropies = cross_entropy_from_probabilities(probabilities, labels)
        correct = predictions == labels

        images_list.append(images.cpu())
        labels_list.append(labels.cpu())
        predictions_list.append(predictions.cpu())
        probabilities_list.append(probabilities.cpu())
        confidences_list.append(confidences.cpu())
        entropies_list.append(entropies.cpu())
        cross_entropies_list.append(cross_entropies.cpu())
        correct_list.append(correct.cpu())
        progress.update(batch_index)

    progress.close()

    return PredictionBundle(
        images=torch.cat(images_list),
        labels=torch.cat(labels_list),
        predictions=torch.cat(predictions_list),
        probabilities=torch.cat(probabilities_list),
        confidences=torch.cat(confidences_list),
        entropies=torch.cat(entropies_list),
        cross_entropies=torch.cat(cross_entropies_list),
        correct=torch.cat(correct_list),
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    return values[mask].mean().item()


def summarize_predictions(bundle: PredictionBundle) -> dict[str, float | int]:
    correct_mask = bundle.correct
    incorrect_mask = ~bundle.correct
    sample_count = bundle.labels.numel()

    return {
        "sample_count": sample_count,
        "correct_count": correct_mask.sum().item(),
        "incorrect_count": incorrect_mask.sum().item(),
        "accuracy": correct_mask.float().mean().item(),
        "average_confidence": bundle.confidences.mean().item(),
        "average_entropy": bundle.entropies.mean().item(),
        "average_cross_entropy": bundle.cross_entropies.mean().item(),
        "correct_average_entropy": masked_mean(bundle.entropies, correct_mask),
        "incorrect_average_entropy": masked_mean(bundle.entropies, incorrect_mask),
        "correct_average_cross_entropy": masked_mean(
            bundle.cross_entropies, correct_mask
        ),
        "incorrect_average_cross_entropy": masked_mean(
            bundle.cross_entropies, incorrect_mask
        ),
        "maximum_possible_entropy": math.log(len(CLASS_NAMES)),
    }


def write_metrics(metrics: dict[str, float | int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.txt"

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Entropy Confidence Lab Summary",
        "================================",
        f"Samples analyzed: {metrics['sample_count']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Average confidence: {metrics['average_confidence']:.4f}",
        f"Average entropy: {metrics['average_entropy']:.4f}",
        f"Average cross-entropy: {metrics['average_cross_entropy']:.4f}",
        "",
        "Correct vs incorrect predictions",
        "--------------------------------",
        f"Correct count: {metrics['correct_count']}",
        f"Incorrect count: {metrics['incorrect_count']}",
        f"Correct average entropy: {metrics['correct_average_entropy']:.4f}",
        f"Incorrect average entropy: {metrics['incorrect_average_entropy']:.4f}",
        (
            "Correct average cross-entropy: "
            f"{metrics['correct_average_cross_entropy']:.4f}"
        ),
        (
            "Incorrect average cross-entropy: "
            f"{metrics['incorrect_average_cross_entropy']:.4f}"
        ),
        "",
        (
            "Reference: maximum entropy for 10 equally likely classes is "
            f"log(10) = {metrics['maximum_possible_entropy']:.4f} nats."
        ),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prediction_csv(bundle: PredictionBundle, output_dir: Path) -> None:
    csv_path = output_dir / "test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "correct",
                "confidence",
                "entropy",
                "cross_entropy",
            ]
        )
        for index in range(bundle.labels.numel()):
            label = bundle.labels[index].item()
            prediction = bundle.predictions[index].item()
            writer.writerow(
                [
                    index,
                    label,
                    CLASS_NAMES[label],
                    prediction,
                    CLASS_NAMES[prediction],
                    bool(bundle.correct[index].item()),
                    f"{bundle.confidences[index].item():.6f}",
                    f"{bundle.entropies[index].item():.6f}",
                    f"{bundle.cross_entropies[index].item():.6f}",
                ]
            )


def write_training_history(
    history: list[dict[str, float | int]], output_dir: Path
) -> None:
    if not history:
        return

    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(
    history: list[dict[str, float | int]], output_dir: Path
) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    test_losses = [row["test_loss"] for row in history]
    train_accuracies = [row["train_accuracy"] for row in history]
    test_accuracies = [row["test_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_losses, marker="o", label="Train loss")
    axes[0].plot(epochs, test_losses, marker="o", label="Test loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train accuracy")
    axes[1].plot(epochs, test_accuracies, marker="o", label="Test accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("Training Progress")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def save_entropy_histogram(bundle: PredictionBundle, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(bundle.entropies.numpy(), bins=40, color="#4C78A8", edgecolor="white")
    plt.axvline(
        bundle.entropies.mean().item(),
        color="#F58518",
        linestyle="--",
        label="Average entropy",
    )
    plt.axvline(
        math.log(len(CLASS_NAMES)),
        color="#54A24B",
        linestyle=":",
        label="Maximum entropy: log(10)",
    )
    plt.title("Entropy Distribution Across Fashion-MNIST Test Predictions")
    plt.xlabel("Entropy H(p) in nats")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_histogram.png", dpi=160)
    plt.close()


def save_correct_vs_incorrect_boxplots(
    bundle: PredictionBundle, output_dir: Path
) -> None:
    correct_mask = bundle.correct
    incorrect_mask = ~bundle.correct

    entropy_groups = [
        bundle.entropies[correct_mask].numpy(),
        bundle.entropies[incorrect_mask].numpy(),
    ]
    cross_entropy_groups = [
        bundle.cross_entropies[correct_mask].numpy(),
        bundle.cross_entropies[incorrect_mask].numpy(),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].boxplot(
        entropy_groups, tick_labels=["Correct", "Incorrect"], showmeans=True
    )
    axes[0].set_title("Entropy by Prediction Result")
    axes[0].set_ylabel("Entropy H(p)")

    axes[1].boxplot(
        cross_entropy_groups,
        tick_labels=["Correct", "Incorrect"],
        showmeans=True,
    )
    axes[1].set_title("Cross-Entropy by Prediction Result")
    axes[1].set_ylabel("-log probability of true class")

    fig.suptitle("Correct vs Incorrect Prediction Uncertainty")
    fig.tight_layout()
    fig.savefig(output_dir / "correct_vs_incorrect_boxplots.png", dpi=160)
    plt.close(fig)


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze(0) * FASHION_MNIST_STD + FASHION_MNIST_MEAN


def top_indices(
    values: torch.Tensor, count: int, largest: bool, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if mask is None:
        candidate_indices = torch.arange(values.numel())
    else:
        candidate_indices = torch.where(mask)[0]

    if candidate_indices.numel() == 0:
        return candidate_indices

    count = min(count, candidate_indices.numel())
    candidate_values = values[candidate_indices]
    selected_positions = torch.topk(candidate_values, k=count, largest=largest).indices
    return candidate_indices[selected_positions]


def format_top_probabilities(probabilities: torch.Tensor, count: int = 3) -> str:
    top_probabilities, top_labels = torch.topk(probabilities, k=count)
    parts = []
    for probability, label in zip(top_probabilities, top_labels):
        class_name = CLASS_NAMES[label.item()]
        parts.append(f"{class_name}: {probability.item():.2f}")
    return "\n".join(parts)


def save_example_grid(
    bundle: PredictionBundle,
    indices: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    if indices.numel() == 0:
        return

    column_count = indices.numel()
    fig, axes = plt.subplots(1, column_count, figsize=(4 * column_count, 4.8))
    if column_count == 1:
        axes = [axes]

    for axis, index_tensor in zip(axes, indices):
        index = index_tensor.item()
        label = bundle.labels[index].item()
        prediction = bundle.predictions[index].item()
        image = denormalize_image(bundle.images[index]).numpy()

        axis.imshow(image, cmap="gray")
        axis.axis("off")
        axis.set_title(
            "\n".join(
                [
                    f"True: {CLASS_NAMES[label]}",
                    f"Pred: {CLASS_NAMES[prediction]}",
                    f"Conf: {bundle.confidences[index].item():.2f}",
                    f"H: {bundle.entropies[index].item():.2f}",
                    f"CE: {bundle.cross_entropies[index].item():.2f}",
                    format_top_probabilities(bundle.probabilities[index]),
                ]
            ),
            fontsize=8,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_example_visualizations(bundle: PredictionBundle, output_dir: Path) -> None:
    example_count = 5
    high_entropy_indices = top_indices(bundle.entropies, example_count, largest=True)
    low_entropy_indices = top_indices(bundle.entropies, example_count, largest=False)
    confident_wrong_indices = top_indices(
        bundle.confidences, example_count, largest=True, mask=~bundle.correct
    )

    save_example_grid(
        bundle,
        high_entropy_indices,
        "High-Entropy Predictions: Uncertain Probability Distributions",
        output_dir / "examples_high_entropy.png",
    )
    save_example_grid(
        bundle,
        low_entropy_indices,
        "Low-Entropy Predictions: Confident Probability Distributions",
        output_dir / "examples_low_entropy.png",
    )
    save_example_grid(
        bundle,
        confident_wrong_indices,
        "Confident Wrong Predictions: Low Uncertainty Can Still Be Wrong",
        output_dir / "examples_confident_wrong.png",
    )


def save_visualizations(bundle: PredictionBundle, output_dir: Path) -> None:
    save_entropy_histogram(bundle, output_dir)
    save_correct_vs_incorrect_boxplots(bundle, output_dir)
    save_example_visualizations(bundle, output_dir)


def main() -> None:
    args = parse_args()
    configure_warning_output(args.debug)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress

    device = choose_device(args.device)
    print(describe_device(device, args.device), flush=True)

    train_loader, test_loader = build_data_loaders(args, device)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    history: list[dict[str, float | int]] = []

    print(
        f"Model: {args.model} | epochs: {args.epochs} | "
        f"batch size: {args.batch_size} | augmentation: {args.augment}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, show_progress
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            label=f"Epoch {epoch:02d}/{args.epochs} test ",
            show_progress=show_progress,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr {current_learning_rate:.6f} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"train acc {train_metrics.accuracy:.4f} | "
            f"test loss {test_metrics.loss:.4f}, "
            f"test acc {test_metrics.accuracy:.4f}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

    prediction_bundle = collect_predictions(model, test_loader, device, show_progress)
    metrics = summarize_predictions(prediction_bundle)
    write_metrics(metrics, args.output_dir)
    write_training_history(history, args.output_dir)
    write_prediction_csv(prediction_bundle, args.output_dir)
    save_training_curves(history, args.output_dir)
    save_visualizations(prediction_bundle, args.output_dir)

    print("\nAnalysis complete.", flush=True)
    print(f"Saved metrics, CSV, and plots to: {args.output_dir}", flush=True)
    print(
        "Key comparison: "
        f"correct entropy={metrics['correct_average_entropy']:.4f}, "
        f"incorrect entropy={metrics['incorrect_average_entropy']:.4f}; "
        f"correct CE={metrics['correct_average_cross_entropy']:.4f}, "
        f"incorrect CE={metrics['incorrect_average_cross_entropy']:.4f}.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(130)
    except Exception as error:
        if "--debug" in sys.argv:
            raise
        print(f"\nError: {sanitize_text(str(error))}")
        print("Run again with --debug to show the full traceback.")
        raise SystemExit(1)
