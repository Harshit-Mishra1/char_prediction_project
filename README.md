# Next Character Prediction (RNN / GRU)

An end-to-end character-level language model built with TensorFlow/Keras. It automatically downloads the Shakespeare text corpus, trains a GRU-based Recurrent Neural Network to predict the next character in a sequence, saves checkpoints after each epoch, and generates new text using temperature sampling.

## Project Structure

```text
char_prediction_project/
├── data/               # Downloaded training text (ignored by Git)
├── models/             # Saved model weight checkpoints (ignored by Git)
├── .gitignore          # Excludes virtual environments, cache, and large files
├── LICENSE             # MIT License
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── train.py            # End-to-end training and text generation script
```

## Model Architecture

1. **Embedding Layer:** Maps character IDs to 256-dimensional dense vectors.
2. **GRU Layer:** 1024-unit Gated Recurrent Unit that tracks sequential character context.
3. **Dense Layer:** Outputs raw logits across the entire character vocabulary.
4. **Temperature Sampling:** Adjusts output logits during inference (`temperature=0.7`) to balance coherence and creativity.

## Setup & Usage (Windows)

1. **Clone the repository**
   ```cmd
   git clone https://github.com/<your-username>/char_prediction_project.git
   cd char_prediction_project
   ```

2. **Create and activate a virtual environment**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run training and text generation**
   ```cmd
   python train.py
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
