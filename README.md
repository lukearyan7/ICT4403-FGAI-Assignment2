# ICT4403 Assignment 2 — Autoregressive Text Generation

## Objective
Implement and compare Simple RNN, Bidirectional RNN, LSTM, and GRU models for next-word prediction and autoregressive text generation. Compare held-out loss, perplexity, accuracy, training time, and generated text at multiple sampling temperatures.

## Dataset
**Tiny Shakespeare** — compact literary text corpus containing excerpts from Shakespeare's plays.

- Project page: https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
- Direct raw text: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The notebook downloads the raw text automatically if it is not already present.

## Models
- **Simple RNN:** recurrent tanh layer with a linear next-token prediction head.
- **Bidirectional RNN:** reads the supplied context in forward and reverse directions.
- **LSTM:** uses input, forget, and output gates to maintain a cell state.
- **GRU:** uses update and reset gates with a more compact gated recurrent design.

All models use the same tokenization, vocabulary, train/validation split, sequence length, embedding/hidden dimensions, optimizer, batch size, and epoch budget for a fair baseline comparison.

## Files
- `ICT4403_Assignment2_Text_Generation.ipynb` — complete implementation, training, evaluation, generation, plots, and analysis prompts.
- `README.md` — project description and execution instructions.
- After running the notebook, it produces `model_comparison.csv`, `training_history.csv`, `model_metrics.png`, `validation_loss.png`, `generated_samples.csv`, and `qualitative_review_template.csv`.

## Requirements
- Python 3.10+
- PyTorch
- NumPy
- pandas
- matplotlib
- requests is not required by the notebook (Python `urllib` is used)

Install dependencies:
```bash
pip install torch numpy pandas matplotlib jupyter
```

## Run
1. Clone or download the repository.
2. Install the dependencies above.
3. Open `ICT4403_Assignment2_Text_Generation.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run all cells from top to bottom.
5. Review the measured metrics and generated samples. Include the actual results in your report; do not use unmeasured or fabricated numbers.

## Evaluation
The notebook reports:
- Validation cross-entropy loss (lower is better)
- Validation perplexity (lower is better)
- Next-word prediction accuracy (higher is better)
- Measured training time (hardware-dependent)

It also generates samples from identical seed strings at temperatures 0.5, 1.0, and 1.5. Lower temperatures typically favor high-probability continuations; higher temperatures usually increase variety but can reduce coherence. These are expected tendencies, and the actual outputs must be inspected.

## Results and conclusion
Run the notebook to populate the results tables and figures. The notebook prints the model with the lowest measured validation perplexity. In the final report, also discuss the generated samples' coherence, contextual consistency, diversity, and repetition, and justify the final model choice using both quantitative and qualitative evidence.

## Experimental limitations
This is a small educational baseline, not a state-of-the-art language model. Training times and metrics depend on hardware. The bidirectional model reads both directions within an observed context window, which is different from strictly causal streaming generation.

## Author
Add your name and registration number in the submission PDF required by the course instructions.
