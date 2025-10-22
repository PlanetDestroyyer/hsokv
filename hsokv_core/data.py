"""Data utilities and synthetic dataset generation for H-SOKV."""

import random
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG


class SimpleTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.vocab: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.inverse_vocab: List[str] = [self.pad_token, self.unk_token]

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    def fit(self, texts: List[str]) -> None:
        for text in texts:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inverse_vocab.append(token)

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = [self.vocab.get(tok, self.unk_token_id) for tok in self._tokenize(text)]
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            tokens += [self.pad_token_id] * (max_length - len(tokens))
        return tokens

    def encode_tokens(self, tokens: List[str], max_length: int) -> List[int]:
        indices = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        indices = indices[:max_length]
        if len(indices) < max_length:
            indices += [self.pad_token_id] * (max_length - len(indices))
        return indices

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace(".", " ").replace(",", " ").replace(";", " ").split()

    def to_dict(self) -> Dict[str, object]:
        return {"vocab": self.inverse_vocab}

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        payload = {"vocab": self.inverse_vocab, "pad_token": self.pad_token, "unk_token": self.unk_token}
        with open(os.path.join(save_directory, "tokenizer.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "SimpleTokenizer":
        path = os.path.join(save_directory, "tokenizer.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tokenizer = cls()
        vocab_list = payload.get("vocab", [])
        tokenizer.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        tokenizer.inverse_vocab = list(vocab_list)
        tokenizer.pad_token = payload.get("pad_token", "<pad>")
        tokenizer.unk_token = payload.get("unk_token", "<unk>")
        return tokenizer

    def __len__(self) -> int:
        return len(self.vocab)


RARE_WORD_SPECS: List[Dict[str, str]] = [
    {"word": "glimmerous", "definition": "shining with intermittent flashes of light", "usage": "The glimmerous lantern flickered warmly in the wind."},
    {"word": "murngale", "definition": "a calm feeling that follows a storm", "usage": "Sailors cherished the murngale that settled after the squall."},
    {"word": "veldrift", "definition": "to wander without direction across open plains", "usage": "Nomads would veldrift for days seeking fresh water."},
    {"word": "thresk", "definition": "a pact sealed with shared silence", "usage": "They formed a thresk that needed no signatures."},
    {"word": "orbelyn", "definition": "a crystalline seed that glows as it sprouts", "usage": "Gardeners planted orbelyn to light the beds at dusk."},
    {"word": "quindle", "definition": "to solve a complex problem through play", "usage": "The team chose to quindle the puzzle with improvisation games."},
    {"word": "saffrine", "definition": "a scent reminiscent of burnt sugar and cedar", "usage": "A saffrine aroma drifted from the market stalls."},
    {"word": "parlune", "definition": "to negotiate via musical phrases", "usage": "Diplomats decided to parlune under the moonlight."},
    {"word": "tressial", "definition": "woven strands that record whispered memories", "usage": "The elder kept a tressial of every promise made."},
    {"word": "gryphel", "definition": "a stubborn mechanical bird powered by steam", "usage": "The gryphel sputtered before leaping into the clouds."},
    {"word": "nimbrel", "definition": "soft rain that carries the smell of fresh stone", "usage": "A nimbrel washed the city's dusty streets."},
    {"word": "harthune", "definition": "the pulsing glow within ancient hearths", "usage": "Even asleep, they felt the harthune steady their breathing."},
    {"word": "skellion", "definition": "a hidden corridor connecting rival libraries", "usage": "Scholars whispered about the skellion beneath the archive."},
    {"word": "weircall", "definition": "a sound that lures fish toward nets", "usage": "Fishers practiced the weircall before dawn."},
    {"word": "imbrasy", "definition": "protective markings drawn with ash", "usage": "Before the festival, children wore imbrasy on their cheeks."},
    {"word": "fenshade", "definition": "mist that obscures boundaries between lands", "usage": "Travellers feared losing their way in the fenshade."},
    {"word": "calith", "definition": "a promise delivered through folded letters", "usage": "She sent a calith to assure him of her return."},
    {"word": "pewther", "definition": "metallic clay used for sculpting echoes", "usage": "Artists shaped pewther into resonant figures."},
    {"word": "lurest", "definition": "the point where light bends into color trails", "usage": "Photographers waited at the lurest after sunset."},
    {"word": "zinthary", "definition": "a cipher that changes with every sunrise", "usage": "Spies swapped zinthary notes to stay ahead."},
]


def build_base_vocabulary() -> List[str]:
    base_words = [
        "the", "village", "forest", "river", "mountain", "wind", "storm", "calm",
        "travelers", "children", "ancient", "stone", "glow", "light", "shadow",
        "music", "dance", "story", "whisper", "promise", "market", "craft",
        "harbor", "ship", "morning", "evening", "twilight", "fire", "breeze",
        "echo", "song", "path", "secret", "hidden", "scholar", "guardian",
        "mystery", "dream", "memory", "silent", "watch", "guide", "soft",
        "bright", "gentle", "storm", "rain", "cloud", "mist", "horizon",
        "valley", "plain", "fields", "stone", "metal", "glow", "fragrant",
        "cedar", "sugar", "scent", "market", "lantern", "garden", "night",
        "moon", "stars", "spark", "steam", "mechanical", "bird", "team",
        "puzzle", "game", "festival", "paint", "door", "archive", "library",
        "fish", "nets", "harbor", "ash", "markings", "travel", "path", "wander",
        "letters", "folded", "artists", "sculpt", "color", "trail", "cipher",
        "dawn", "sunrise", "promise", "return",
    ]
    more_words = ["journey", "circle", "gather", "create", "listen", "learn", "remember", "share"]
    base_words.extend(more_words)
    return base_words


def random_sentence(words: List[str], length: int) -> str:
    return " ".join(random.choices(words, k=length))


def build_story(rare_spec: Dict[str, str], base_vocab: List[str], distractors: List[str]) -> str:
    lead = random_sentence(base_vocab, random.randint(7, 12))
    middle = f"{rare_spec['word']} means {rare_spec['definition']}."
    usage = rare_spec["usage"]
    distractor = random_sentence(distractors, random.randint(6, 10))
    return " ".join([lead, middle, usage, distractor])


def generate_dataset(
    tokenizer: Optional[SimpleTokenizer] = None,
    fit_tokenizer: bool = True,
) -> Tuple[Dict[str, List[Dict[str, object]]], SimpleTokenizer, Dict[str, int]]:
    base_vocab = build_base_vocabulary()
    distractors = base_vocab + ["ordinary", "common", "simple", "banal", "routine"]
    examples: Dict[str, List[Dict[str, object]]] = {"train": [], "test": [], "retention": [], "distractor": []}
    all_texts: List[str] = []
    word_counts: Dict[str, int] = {}
    for idx, spec in enumerate(RARE_WORD_SPECS):
        train_count = random.randint(1, 5)
        word_counts[spec["word"]] = train_count
        for _ in range(train_count):
            story = build_story(spec, base_vocab, distractors)
            all_texts.append(story)
            entry = {
                "story": story,
                "rare_word": spec["word"],
                "definition": spec["definition"],
                "usage": spec["usage"],
                "word_id": idx,
                "num_examples": train_count,
            }
            examples["train"].append(entry)
        for _ in range(5):
            story = build_story(spec, base_vocab, distractors)
            all_texts.append(story)
            examples["test"].append(
                {
                    "story": story,
                    "rare_word": spec["word"],
                    "definition": spec["definition"],
                    "usage": spec["usage"],
                    "word_id": idx,
                    "num_examples": train_count,
                }
            )
        for _ in range(3):
            story = build_story(spec, base_vocab, distractors)
            all_texts.append(story)
            examples["retention"].append(
                {
                    "story": story,
                    "rare_word": spec["word"],
                    "definition": spec["definition"],
                    "usage": spec["usage"],
                    "word_id": idx,
                    "num_examples": train_count,
                }
            )
    for _ in range(len(RARE_WORD_SPECS) * CONFIG["retention_distractor_factor"]):
        story = random_sentence(distractors, random.randint(12, 18))
        all_texts.append(story)
        examples["distractor"].append({"story": story})
    tokenizer = tokenizer or SimpleTokenizer()
    if fit_tokenizer:
        tokenizer.fit(all_texts + [spec["definition"] for spec in RARE_WORD_SPECS] + [spec["usage"] for spec in RARE_WORD_SPECS])
    return examples, tokenizer, word_counts


class RareWordDataset(Dataset):
    def __init__(self, examples: List[Dict[str, object]], tokenizer: SimpleTokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.examples[idx]
        tokens = self.tokenizer.encode(sample["story"], self.max_length)
        label = sample.get("word_id", -1)
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "rare_word": sample.get("rare_word", ""),
            "definition": sample.get("definition", ""),
            "usage": sample.get("usage", ""),
            "num_examples": sample.get("num_examples", 0),
            "word_id": sample.get("word_id", -1),
        }


def build_collate(tokenizer: SimpleTokenizer):
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = torch.stack([item["label"] for item in batch])
        rare_words = [item["rare_word"] for item in batch]
        definitions = [item["definition"] for item in batch]
        usages = [item["usage"] for item in batch]
        num_examples = torch.tensor([item["num_examples"] for item in batch], dtype=torch.long)
        word_ids = torch.tensor([item["word_id"] for item in batch], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rare_words": rare_words,
            "definitions": definitions,
            "usages": usages,
            "num_examples": num_examples,
            "word_ids": word_ids,
        }

    return collate_fn


def prepare_dataloaders(
    dataset: Dict[str, List[Dict[str, object]]],
    tokenizer: SimpleTokenizer,
    config: Dict[str, object],
) -> Dict[str, DataLoader]:
    collate = build_collate(tokenizer)
    loader_kwargs = {
        "pin_memory": config.get("pin_memory", False),
        "num_workers": config.get("num_workers", 0),
    }

    if dataset.get("val"):
        train_examples = list(dataset["train"])
        val_examples = list(dataset["val"])
    else:
        grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for sample in dataset["train"]:
            grouped[sample["word_id"]].append(sample)
        train_examples = []
        val_examples = []
        for samples in grouped.values():
            if len(samples) > 1:
                val_examples.append(samples.pop())
            train_examples.extend(samples)

    train_loader = DataLoader(
        RareWordDataset(train_examples, tokenizer, config["max_seq_length"]),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        RareWordDataset(val_examples, tokenizer, config["max_seq_length"]),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        RareWordDataset(dataset["test"], tokenizer, config["max_seq_length"]),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )
    retention_loader = DataLoader(
        RareWordDataset(dataset["retention"], tokenizer, config["max_seq_length"]),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "retention_loader": retention_loader,
    }


def _load_corpus_text(corpus_path: Optional[str]) -> str:
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as handle:
            return handle.read()
    fallback_passages = [
        "Language models estimate the probability of text sequences by conditioning on the tokens that came before.",
        "Transformers attend over all positions but their quadratic cost motivates memory-augmented replacements.",
        "Hierarchical swarm optimization explores optimizer choices in parallel while KV memories cache facts.",
        "We test the architecture on a tiny corpus so that experiments fit inside quick development loops.",
    ]
    return "\n".join(fallback_passages)


def generate_language_model_dataset(
    config: Dict[str, object],
    corpus_path: Optional[str] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> Tuple[Dict[str, List[Dict[str, object]]], SimpleTokenizer, Dict[str, int], List[str]]:
    seq_length = int(config.get("lm_seq_length", config.get("max_seq_length", 96)))
    stride = int(config.get("lm_stride", max(1, seq_length // 2)))
    max_sequences = int(config.get("lm_max_sequences", 20000))
    train_split = float(config.get("lm_train_split", 0.8))
    val_split = float(config.get("lm_val_split", 0.1))
    text = _load_corpus_text(corpus_path or config.get("lm_corpus_path"))
    if not text.strip():
        raise ValueError("Language modeling corpus is empty. Provide --lm-corpus with non-empty text.")
    tokenizer = tokenizer or SimpleTokenizer()
    if len(tokenizer.vocab) <= 2:
        tokenizer.fit([text])
    tokens = tokenizer._tokenize(text)
    if len(tokens) <= seq_length + 1:
        raise ValueError(
            f"Corpus too short for seq_length={seq_length}. Provide a longer corpus or reduce --lm-seq-length."
        )
    samples: List[Dict[str, object]] = []
    for start in range(0, len(tokens) - seq_length - 1, max(1, stride)):
        context_tokens = tokens[start : start + seq_length]
        target_token = tokens[start + seq_length]
        story = " ".join(context_tokens)
        label_id = tokenizer.vocab.get(target_token, tokenizer.unk_token_id)
        sample = {
            "story": story,
            "rare_word": target_token,
            "definition": f"next token {target_token}",
            "usage": target_token,
            "word_id": label_id,
            "num_examples": 1,
        }
        samples.append(sample)
        if len(samples) >= max_sequences:
            break
    if len(samples) < 10:
        raise ValueError(
            f"Only generated {len(samples)} language modeling samples. Increase corpus size or adjust stride."
        )
    train_end = max(1, int(len(samples) * train_split))
    val_end = min(len(samples), max(train_end + 1, int(len(samples) * (train_split + val_split))))
    train_split_samples = samples[:train_end]
    val_split_samples = samples[train_end:val_end]
    test_split_samples = samples[val_end:] if val_end < len(samples) else samples[train_end:]
    if not val_split_samples and train_split_samples:
        val_split_samples = [train_split_samples.pop()]
    if not test_split_samples and val_split_samples:
        test_split_samples = [val_split_samples[-1]]
    word_counts = Counter(sample["rare_word"] for sample in train_split_samples)
    for sample in train_split_samples:
        sample["num_examples"] = word_counts[sample["rare_word"]]
    retention_size = min(len(test_split_samples), max(1, len(test_split_samples) // 5))
    retention_samples = list(test_split_samples[:retention_size])
    dataset = {
        "train": train_split_samples,
        "val": val_split_samples,
        "test": test_split_samples,
        "retention": retention_samples,
        "distractor": [],
    }
    rng = random.Random(config.get("seed", 42))
    distractor_count = min(32, max(4, len(train_split_samples) // 10))
    vocab_tokens = [tok for tok in tokenizer.inverse_vocab if tok not in (tokenizer.pad_token, tokenizer.unk_token)]
    if vocab_tokens:
        for _ in range(distractor_count):
            noise_tokens = [rng.choice(vocab_tokens) for _ in range(seq_length)]
            dataset["distractor"].append({"story": " ".join(noise_tokens)})
    label_names = list(tokenizer.inverse_vocab)
    label_name_counts = {token: word_counts.get(token, 0) for token in label_names}
    return dataset, tokenizer, label_name_counts, label_names
