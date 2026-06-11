# Entropy Confidence Lab

This project explores how entropy and cross-entropy help us understand model confidence, uncertainty, and prediction quality in machine learning.

A simple classifier is trained on Fashion-MNIST, then its predictions are analyzed using manually implemented entropy and cross-entropy calculations.

## Goals

- Train a simple classification model on Fashion-MNIST
- Compute prediction entropy manually
- Implement cross-entropy loss manually
- Compare entropy and loss for correct vs incorrect predictions
- Visualize confident, uncertain, correct, and incorrect examples
- Connect the experiment to information theory concepts

## Concepts Covered

- Probability distributions
- Entropy
- Cross-entropy
- KL divergence
- Model confidence
- Maximum likelihood training
- Softmax probabilities

## Dataset

This project uses Fashion-MNIST, a 10-class image classification dataset containing grayscale clothing images.

Classes include:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Planned Outputs

The project will generate:

- Training and test accuracy
- Average entropy for correct vs incorrect predictions
- Average cross-entropy for correct vs incorrect predictions
- Histogram of entropy values
- Box plots comparing correct and incorrect predictions
- Example images with prediction probabilities, entropy, and cross-entropy values

Generated files are saved in `outputs/`:

- `metrics.json`
- `summary.txt`
- `training_history.csv`
- `training_curves.png`
- `test_predictions.csv`
- `entropy_histogram.png`
- `correct_vs_incorrect_boxplots.png`
- `examples_high_entropy.png`
- `examples_low_entropy.png`
- `examples_confident_wrong.png`

## Why This Matters

Accuracy alone does not tell us how confident a model is.

Entropy helps measure uncertainty in the model’s predicted probability distribution. Cross-entropy measures how much probability the model assigned to the correct class. Together, they help reveal when a model is confident, uncertain, correct, or confidently wrong.

## Tech Stack

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Train the model and generate the analysis:

```bash
python main.py
```

The default model is a small CNN. For a stronger full run, use:

```bash
python main.py --model cnn --epochs 15 --batch-size 128 --augment
```

The script defaults to `--num-workers 0` so errors and warnings stay readable.
If you want faster data loading on your own machine, you can try:

```bash
python main.py --num-workers 2
```

Device selection defaults to `--device auto`. If CUDA is available, the script
uses the GPU; otherwise it automatically falls back to CPU. To force CPU even on
a machine with a GPU, run:

```bash
python main.py --device cpu
```

For a faster smoke test, use fewer samples and one epoch:

```bash
python main.py --epochs 1 --train-limit 2000 --test-limit 500
```

Disable progress bars if you are redirecting output:

```bash
python main.py --no-progress
```

By default, warnings and runtime errors are displayed without local file paths.
If you need the full traceback for debugging, run:

```bash
python main.py --debug
```

## Implementation Notes

The project intentionally avoids built-in entropy and cross-entropy loss functions.

- Training uses a custom cross-entropy function from logits:

```text
log_softmax(z_i) = z_i - log(Σ exp(z_j))
loss = -log_softmax(z_true)
```

- Entropy is computed manually from prediction probabilities:

```text
H(p) = -Σ p(x) log p(x)
```

- Cross-entropy for each test prediction is computed manually:

```text
CE = -log(probability assigned to the true class)
```

This makes the project focus on understanding the math instead of hiding it behind library calls.
