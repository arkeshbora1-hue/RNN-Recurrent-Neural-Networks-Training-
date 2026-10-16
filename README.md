# Recurrent Neural Networks: From First Experiment to Reusable Toolkit

A hands-on Keras/TensorFlow notebook that starts with a single SimpleRNN learning to count, tests where that model fails, and then builds a reusable RNN toolkit. The toolkit is applied to sequence-rule learning, time-series forecasting, Shakespeare-style text generation, and an encoder-decoder, plus a look inside the network's hidden state.

**Notebook:** `RNN_Recurrent_Neural_Networks_.ipynb`
**Runs in:** Google Colab (a T4 GPU is recommended for the text-generation section)

---

## Highlights

| Experiment | Model | Result |
|---|---|---|
| Sequence rules on unseen data | LSTM + window-relative scaling | Held-out MAE **0.005** |
| Out-of-range probe `[5000, 5004, 5008]` | same | Predicts **5012.01** (correct: 5012) |
| 12-step time-series forecast | SimpleRNN / GRU / LSTM | Test MAE **0.59 / 0.67 / 0.67**, vs **1.24** for the seasonal-naive baseline |
| Character-level text generation | GRU (256 units) | Generates Shakespeare-style dialogue with speaker names and verse layout |
| English → French translation | MarianMT (pretrained Transformer) | *"I love deep learning"* → *"J'adore l'apprentissage profond"* |

---

## What's inside

The notebook has two stages.

### Stage 1: First experiments (cells 0 to 11)

1. **Counting RNN.** A `SimpleRNN(20)` is trained on windows `[i, i+1, i+2] → i+3` for 500 epochs and predicts `[10, 11, 12] → 13.0`.
2. **Translation demo.** The pretrained `Helsinki-NLP/opus-mt-en-fr` model from Hugging Face translates English sentences to French.
3. **Train/test split** of the counting data with scikit-learn.

**What this stage taught.** When tested on sequences it had never seen, the counting model had only learned "add about 1". It predicted roughly 11 for `[5, 8, 11]` (correct: 14) and roughly 18 for `[20, 18, 16]` (correct: 14). Its one correct test, `[10, 11, 12]`, was a row from the training data. The translation model is a Transformer, not an RNN. Stage 2 fixes each of these problems.

### Stage 2: The RNN toolkit and applications (cells 12 to 23)

**Toolkit (cells 12 to 14)**

| Function | Purpose |
|---|---|
| `make_windows(series, window, horizon)` | Slides a window along any 1-D or multi-feature series to create `(X, y)` training pairs |
| `chronological_split(X, y)` | Splits into train, validation and test **in time order**, so no future data leaks into training |
| `RelativeScaler` | Expresses each window relative to its last value, so the model learns *changes* and works at any magnitude |
| `build_rnn(task, ...)` | One builder for every task. It picks the right output layer and loss automatically |
| `fit(...)` | Trains with early stopping and learning-rate reduction |
| `plot_history(h)` | Plots training against validation loss and accuracy |

`build_rnn` options:

| Argument | Values |
|---|---|
| `task` | `"regression"`, `"binary"`, `"multiclass"`, `"language_model"` |
| `cell` | `"simple"`, `"lstm"`, `"gru"` |
| `units` | tuple of layer sizes, e.g. `(64,)` or `(128, 64)` for stacked layers |
| `bidirectional` | `True` reads the sequence in both directions |
| `vocab_size` | set it for token inputs; adds an `Embedding` layer |

Gradient clipping (`clipnorm=1.0`) is built in to guard against exploding gradients.

**Applications**

| Cells | Application | RNN pattern |
|---|---|---|
| 15 to 16 | **Sequence rules.** Arithmetic sequences with random starts and steps from −5 to +5, then probes on unseen and out-of-range inputs | many-to-one regression |
| 17 to 19 | **Time-series forecasting.** A synthetic signal with trend, daily and weekly cycles and noise; predicts the next 12 steps from the last 48, and compares SimpleRNN, GRU and LSTM against a seasonal-naive baseline | many-to-many (multi-step) regression |
| 20 to 21 | **Text generation.** A character-level GRU trained on 400,000 characters of Shakespeare, sampled with temperature | language model |
| 22 | **Encoder-decoder.** An LSTM encoder compresses a digit sequence into its final state; an LSTM decoder writes it back in reverse. This is the RNN predecessor of the Transformer used in Stage 1 | sequence-to-sequence |
| 23 | **Hidden-state heatmap.** Records the forecaster's hidden state at every time step and plots it, showing units that track the signal's rhythm | interpretation |

---

## Results in detail

**Sequence rules (cell 16)**

| Input | Correct | Stage 1 model | Stage 2 model |
|---|---|---|---|
| `[10, 11, 12]` | 13 | 13.0 | 12.99 |
| `[500, 501, 502]` | 503 | 504 to 507 | 502.99 |
| `[5, 8, 11]` | 14 | ≈ 11 | 14.00 |
| `[20, 18, 16]` | 14 | ≈ 18 | 14.01 |
| `[5000, 5004, 5008]` | 5012 | not tested | 5012.01 |

**Forecasting (cell 18).** All three RNNs roughly halve the baseline's error. SimpleRNN comes out slightly ahead on this data, because the pattern repeats inside the 48-step window and relative scaling makes the input easy. Gated cells (LSTM, GRU) pay off when information has to survive many more steps.

**Text generation (cell 21).** At temperature 0.8 the model produces correctly formatted play dialogue (`BUCKINGHAM:`, `KING RICHARD III:`) with plausible rhythm and mostly real words. Some invented words remain, which is typical for a character-level model of this size.

---

## Quick start

1. Open the notebook in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**).
2. Optionally select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order (**Runtime → Run all**).

The Shakespeare corpus downloads automatically. If the download fails, a short built-in passage is used so the cell still runs.

### Requirements

Everything except the translation libraries is preinstalled in Colab.

```
tensorflow >= 2.16   (Keras 3)
numpy
matplotlib
scikit-learn
transformers, sentencepiece   # translation demo only; installed by cell 7
```

To run locally:

```bash
pip install tensorflow numpy matplotlib scikit-learn transformers sentencepiece
jupyter notebook RNN_Recurrent_Neural_Networks_.ipynb
```

---

## Using the toolkit on your own data

```python
# Forecast the next 24 values of your own series
import pandas as pd
series = pd.read_csv("my_data.csv")["value"].to_numpy()

X, y = make_windows(series, window=96, horizon=24)
(Xtr, ytr), (Xva, yva), (Xte, yte) = chronological_split(X, y)

scaler = RelativeScaler().fit(Xtr)
Xtr_s, ytr_s = scaler.transform(Xtr, ytr)
Xva_s, yva_s = scaler.transform(Xva, yva)

model = build_rnn("regression", (96, 1), cell="gru", units=(64,), n_outputs=24)
fit(model, Xtr_s, ytr_s, (Xva_s, yva_s), max_epochs=100)

forecast = scaler.inverse_y(Xte, model.predict(scaler.transform(Xte)))
```

```python
# Classify padded token sequences (e.g. reviews) as positive or negative
model = build_rnn("binary", cell="lstm", units=(64,), bidirectional=True, vocab_size=10_000)
```

---

## Known issues

- **`build_rnn` default for `units`.** It is written as `units=(65)`. In Python, `(65)` is the integer 65, not a tuple, so calling `build_rnn` *without* `units` raises `TypeError: 'int' object is not iterable`. Every call in the notebook passes `units` explicitly, so the notebook runs. Change the default to `units=(64,)` (note the comma) to fix it.
- **Cells 22 and 23 have not been run yet** in the saved notebook, so their outputs are empty. In a test run, the encoder-decoder reached **100% exact-match accuracy** on 500 unseen sequences.
- **Variable shadowing.** Cell 8 assigns the translation model to `model`, which replaces the counting RNN from cell 4. Re-run cells 4 to 5 if you need the RNN again, or rename the translation model to `translator`.
- **Duplicate reshape.** Cells 2 and 3 apply the same reshape twice. This is harmless but redundant.

---

## Next steps

1. **Add attention** to the encoder-decoder, then try longer inputs (`SRC_LEN = 20`) to see why attention was needed. This leads directly to the Transformer used in the translation demo.
2. **Add sentiment classification** on the IMDB dataset using `build_rnn("binary", ..., vocab_size=10_000, bidirectional=True)`.
3. **Run the forecaster on real data**, such as energy load, weather or sales, with extra input features.
4. **Tune hyperparameters** with `keras_tuner` over `cell`, `units`, `dropout` and `lr`.
5. **Compare against a small Transformer** on the text-generation task.

---

## Key concepts covered

Hidden state and recurrence · vanishing and exploding gradients · SimpleRNN vs LSTM vs GRU · many-to-one, many-to-many and sequence-to-sequence patterns · teacher forcing · temperature sampling · chronological splitting and data leakage · window-relative scaling · baselines · early stopping
