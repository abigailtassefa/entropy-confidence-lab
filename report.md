# Entropy Confidence Lab Report

## Objective

The goal of this project was to understand entropy, cross-entropy, model confidence, and uncertainty through a practical Fashion-MNIST classification experiment. I trained a CNN classifier, manually implemented the key information-theory calculations, and analyzed how the model behaves when it is correct, incorrect, confident, and uncertain.

The central question was not only “how accurate is the model?” but also “how confident is the model, and does that confidence always mean it is correct?”

## PDF Key Questions Covered

This report explicitly answers the key questions from the task PDF:

- What entropy or information means: see “Information Theory Concepts.”
- What entropy tells about the model: see “What Entropy Tells About the Model.”
- What high and low entropy mean: see “What Entropy Tells About the Model” and “High-Entropy and Low-Entropy Examples.”
- Whether confidence always means correctness: see “Confidence Does Not Always Mean Correctness.”
- What I learned from the experiment: see “What I Learned.”
- Simple formula proof or explanation: see “Formula Explanation.”

## Dataset and Model

The experiment uses Fashion-MNIST, a 10-class grayscale image dataset containing clothing items such as shirts, trousers, coats, bags, sandals, sneakers, and ankle boots.

The final training run used:

- Dataset: full Fashion-MNIST training set and full test set
- Model: convolutional neural network
- Optimizer: AdamW
- Learning-rate schedule: cosine annealing
- Epochs: 10
- Batch size: 512
- Device: NVIDIA GeForce RTX 4070 Laptop GPU
- Final test samples analyzed: 10,000

Run command:

```bash
python main.py --model cnn --epochs 10 --batch-size 512 --learning-rate 0.001 --weight-decay 0.0001 --no-download --no-progress --device cuda
```

## Information Theory Concepts

Information theory gives a mathematical way to talk about surprise and uncertainty.

For an event with probability `p`, the information or surprise is:

```text
I(x) = -log p(x)
```

If an event is very likely, `p` is close to `1`, so `-log(p)` is small. If an event is unlikely, `p` is close to `0`, so `-log(p)` is large. This is why rare events are more “surprising.”

Entropy is the average surprise of a probability distribution:

```text
H(p) = -Σ p(x) log p(x)
```

In this project, the distribution `p` is the model’s softmax output over the 10 Fashion-MNIST classes. Low entropy means the model placed most probability on one class. High entropy means the probability mass was spread across multiple classes.

For 10 classes, the maximum possible entropy is:

```text
log(10) = 2.3026 nats
```

This maximum occurs when the model is completely uncertain and assigns equal probability to all classes.

## Cross-Entropy and Maximum Likelihood

Cross-entropy compares the true label distribution `p` with the model’s predicted distribution `q`:

```text
H(p, q) = -Σ p(x) log q(x)
```

For classification, the true label is one-hot. That means only the true class has probability `1`, and all other classes have probability `0`. So cross-entropy becomes:

```text
loss = -log(probability assigned to the true class)
```

This loss is small when the model assigns high probability to the correct class and large when the model assigns low probability to the correct class.

Training with cross-entropy is maximum likelihood training. The model parameters are adjusted to make the observed training labels as probable as possible:

```text
maximize Π P(model assigns correct label | image)
```

Because products of probabilities are hard to optimize directly, we maximize the log-likelihood instead:

```text
maximize Σ log P(correct label | image)
```

Machine learning optimizers usually minimize losses, so we minimize the negative log-likelihood:

```text
minimize -Σ log P(correct label | image)
```

For classification, this is exactly cross-entropy loss.

## KL Divergence Connection

Entropy, cross-entropy, and KL divergence are connected:

```text
H(p, q) = H(p) + D_KL(p || q)
```

KL divergence measures how different the model distribution `q` is from the true distribution `p`. Since the true labels are fixed, minimizing cross-entropy also minimizes the KL divergence between the true labels and the model’s predicted probabilities.

In simple terms: cross-entropy trains the model to move its predicted probability distribution closer to the true label distribution.

## Formula Explanation

The entropy formula:

```text
H(p) = -Σ p(x) log p(x)
```

comes from taking the expected value of information. If information/surprise is:

```text
I(x) = -log p(x)
```

then the average surprise over all outcomes is:

```text
E[I(x)] = Σ p(x) I(x)
        = Σ p(x)(-log p(x))
        = -Σ p(x) log p(x)
```

This is entropy.

For cross-entropy:

```text
H(p, q) = -Σ p(x) log q(x)
```

In classification, the true label distribution `p` is one-hot. If the true class is `y`, then:

```text
p(y) = 1
p(other classes) = 0
```

So the full sum collapses to only the true class:

```text
H(p, q) = -1 · log q(y)
        = -log q(y)
```

That is why the classification loss becomes:

```text
loss = -log(probability assigned to the true class)
```

## Final Results

The final model achieved the target of more than 90% accuracy.

| Metric | Value |
|---|---:|
| Test accuracy | 91.79% |
| Correct predictions | 9,179 / 10,000 |
| Incorrect predictions | 821 / 10,000 |
| Average confidence | 0.9262 |
| Average entropy | 0.1981 |
| Average cross-entropy | 0.2245 |
| Maximum possible entropy | 2.3026 |

Training improved steadily over 10 epochs:

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.5437 | 80.42% | 0.3610 | 86.49% |
| 2 | 0.3198 | 88.68% | 0.2971 | 89.11% |
| 3 | 0.2743 | 89.99% | 0.2758 | 89.80% |
| 4 | 0.2487 | 90.97% | 0.2635 | 90.18% |
| 5 | 0.2265 | 91.68% | 0.2514 | 91.08% |
| 6 | 0.2053 | 92.61% | 0.2389 | 91.16% |
| 7 | 0.1930 | 93.09% | 0.2332 | 91.43% |
| 8 | 0.1785 | 93.54% | 0.2307 | 91.48% |
| 9 | 0.1694 | 93.82% | 0.2256 | 91.73% |
| 10 | 0.1649 | 94.06% | 0.2245 | 91.79% |

The generated training curve is saved at:

```text
outputs/training_curves.png
```

![Training curves](outputs/training_curves.png)

## Correct vs Incorrect Predictions

The most important part of the experiment was comparing correct and incorrect predictions.

| Group | Average Entropy | Average Cross-Entropy |
|---|---:|---:|
| Correct predictions | 0.1498 | 0.0619 |
| Incorrect predictions | 0.7383 | 2.0423 |

Correct predictions had much lower entropy and much lower cross-entropy. This means that when the model was correct, it usually assigned most probability to one class and assigned very high probability to the true class.

Incorrect predictions had higher entropy on average, meaning the model was often more uncertain when it made mistakes. Their cross-entropy was also much higher because the probability assigned to the true class was low.

The box plot comparing these groups is saved at:

```text
outputs/correct_vs_incorrect_boxplots.png
```

![Correct vs incorrect boxplots](outputs/correct_vs_incorrect_boxplots.png)

## What Entropy Tells About the Model

Entropy tells us how spread out the model’s predicted probabilities are.

Low entropy means:

- the model is confident;
- one class has most of the probability;
- the model has little uncertainty.

High entropy means:

- the model is uncertain;
- several classes have similar probabilities;
- the image may be ambiguous or visually similar to multiple classes.

The average entropy was only `0.1981`, far below the maximum possible entropy of `2.3026`. This shows that the trained CNN was usually confident on the Fashion-MNIST test set.

The entropy histogram is saved at:

```text
outputs/entropy_histogram.png
```

![Entropy histogram](outputs/entropy_histogram.png)

## Confidence Does Not Always Mean Correctness

One of the most important lessons is that a confident prediction is not always correct.

The model made 821 incorrect predictions. Among those wrong predictions:

- 113 wrong predictions had confidence at least 0.90.
- 63 wrong predictions had confidence at least 0.95.
- 24 wrong predictions had entropy below 0.10.

The most confident wrong example was:

```text
True class: Coat
Predicted class: Dress
Confidence: 0.9998
Entropy: 0.0021
Cross-entropy: 8.6165
```

This is a perfect example of why entropy and cross-entropy measure different things. The entropy was extremely low because the model was very confident. But the cross-entropy was extremely high because the model was confidently wrong and assigned very little probability to the true class.

The confident wrong examples are saved at:

```text
outputs/examples_confident_wrong.png
```

![Confident wrong predictions](outputs/examples_confident_wrong.png)

## High-Entropy and Low-Entropy Examples

High-entropy examples usually correspond to ambiguous cases. For example, the highest-entropy example had confidence only `0.3907`, meaning the model did not strongly prefer one class.

Low-entropy examples were usually easy cases where the model predicted almost perfectly. Several low-entropy examples had confidence near `1.0`, entropy near `0.0`, and cross-entropy near `0.0`.

Example grids are saved at:

```text
outputs/examples_high_entropy.png
outputs/examples_low_entropy.png
```

![High-entropy examples](outputs/examples_high_entropy.png)

![Low-entropy examples](outputs/examples_low_entropy.png)

## What I Learned

This task made the difference between accuracy, entropy, and cross-entropy much clearer.

Accuracy only says whether the final predicted class was correct. It does not show how confident the model was.

Entropy describes uncertainty in the model’s predicted probability distribution. It does not use the true label, so it cannot directly say whether the model is right or wrong.

Cross-entropy uses the true label. It measures how much probability the model assigned to the correct class, so it is directly useful as a training loss.

The most interesting discovery was that low entropy can happen for both correct and incorrect predictions. A model can be confidently correct, but it can also be confidently wrong. This is why model confidence should be analyzed carefully, especially in high-stakes applications.

## Conclusion

The final CNN achieved `91.79%` test accuracy with an average cross-entropy loss of `0.2245`. The entropy and cross-entropy analysis showed that correct predictions were usually confident and low-loss, while incorrect predictions had higher uncertainty and much higher loss.

The experiment also showed the limitation of confidence: low entropy does not guarantee correctness. Cross-entropy is more directly tied to learning because it rewards the model for assigning high probability to the true class and penalizes it heavily when the true class receives low probability.

Overall, entropy helped explain uncertainty, cross-entropy explained prediction quality, and their comparison made the model’s behavior much more interpretable than accuracy alone.
