"""
Generates a modest, varied placeholder text corpus so we can train a
REAL, structurally-correct BPE tokenizer (not fake/hand-written files).

IMPORTANT: this is sample/bootstrap text, not your project's real data.
Re-run tokenizer/train_tokenizer.py on your actual data/cleaned/ corpus
once you have real data collected, to get a proper 32K-vocab tokenizer.
"""
import random

random.seed(42)

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Machine learning models learn patterns from large amounts of data.",
    "A transformer architecture uses self-attention to process sequences.",
    "The small language model was trained from scratch on a custom corpus.",
    "Tokenization splits raw text into subword units for the model to consume.",
    "Deep learning has transformed natural language processing in recent years.",
    "The researchers collected, cleaned, and deduplicated the training data.",
    "Byte pair encoding merges frequent character pairs into single tokens.",
    "Attention mechanisms allow models to focus on relevant parts of input.",
    "Pretraining on diverse text helps the model generalize to new tasks.",
    "The optimizer updates model weights using gradients computed from loss.",
    "RMSNorm and RoPE are common components in modern transformer models.",
    "Supervised fine-tuning adapts a base model to follow instructions.",
    "Preference optimization aligns model outputs with human judgments.",
    "def train_model(config): return build_transformer(config.vocab_size)",
    "import torch; model = TransformerModel(layers=12, heads=8, dim=768)",
    "class Tokenizer: def encode(self, text): return self.bpe.encode(text)",
    "The dataset was split into training, validation, and test partitions.",
    "Quantization reduces model precision to speed up inference on edge devices.",
    "The chef carefully prepared a delicious meal using fresh local ingredients.",
    "Scientists discovered a new species of butterfly in the tropical rainforest.",
    "The stock market experienced significant volatility throughout the week.",
    "Students gathered in the library to study for their upcoming examinations.",
    "The mountain climbers reached the summit just before sunrise this morning.",
    "Renewable energy sources are becoming increasingly cost-effective worldwide.",
    "The novelist spent years crafting an intricate plot for her latest book.",
    "Engineers designed a bridge capable of withstanding severe earthquakes.",
    "The orchestra performed a beautiful symphony to a sold-out audience.",
    "Farmers rely on accurate weather forecasts to plan their planting schedules.",
    "The museum unveiled a new exhibit featuring ancient Egyptian artifacts.",
]

def main(out_path: str, repeats: int = 400):
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(repeats):
            random.shuffle(SENTENCES)
            for s in SENTENCES:
                f.write(s + "\n")

if __name__ == "__main__":
    main("/home/claude/my-slm/tokenizer_bootstrap/sample_corpus.txt")
    print("wrote sample corpus")
