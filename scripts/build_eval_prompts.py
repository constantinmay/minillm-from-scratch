"""Generate fixed evaluation prompts covering 5 types.

Types:
  1. keyword_story (40): "Write a short story using these words: X, Y, Z."
  2. topic_story   (20): "Write a happy/sad/funny story about X."
  3. continue_story(20): "Continue the story: [opening sentence]"
  4. style_control (10): "Write a [happy/sad/funny] story about X."
  5. format_control(10): "Write a story in exactly three sentences about X."

Total: 100 prompts

Usage:
  python scripts/build_eval_prompts.py \
      --output data/prompts/eval_prompts.jsonl \
      --num_prompts 100
"""

import argparse
import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# Word banks for keyword_story prompts
# ---------------------------------------------------------------------------

_WORD_SETS = [
    ["cat", "garden", "sunshine"],
    ["dog", "ball", "happiness"],
    ["bird", "nest", "morning"],
    ["fish", "pond", "reflection"],
    ["star", "wish", "midnight"],
    ["flower", "rain", "rainbow"],
    ["moon", "owl", "silence"],
    ["bear", "cave", "winter"],
    ["rabbit", "carrot", "spring"],
    ["cloud", "wind", "freedom"],
    ["tree", "leaf", "autumn"],
    ["river", "stone", "patience"],
    ["butterfly", "meadow", "dance"],
    ["snow", "fireplace", "warmth"],
    ["ocean", "shell", "discovery"],
    ["mountain", "eagle", "courage"],
    ["forest", "path", "wonder"],
    ["island", "treasure", "map"],
    ["castle", "dragon", "friendship"],
    ["village", "bell", "celebration"],
    ["garden", "seed", "growth"],
    ["kitten", "yarn", "play"],
    ["puppy", "mud", "laugh"],
    ["squirrel", "acorn", "preparation"],
    ["penguin", "ice", "journey"],
    ["whale", "song", "deep"],
    ["horse", "freedom", "field"],
    ["lighthouse", "storm", "bravery"],
    ["library", "book", "adventure"],
    ["kitchen", "recipe", "surprise"],
    ["fox", "grape", "clever"],
    ["frog", "lily", "leap"],
    ["spider", "web", "patience"],
    ["caterpillar", "leaf", "change"],
    ["deer", "fawn", "gentle"],
    ["elephant", "memory", "wise"],
    ["giraffe", "sky", "tall"],
    ["parrot", "pirate", "talk"],
    ["turtle", "race", "slow"],
    ["dolphin", "wave", "joy"],
]

# ---------------------------------------------------------------------------
# Topics for topic_story / style_control / format_control
# ---------------------------------------------------------------------------

_TOPICS = [
    "a lost puppy finding its way home",
    "a magical tree in the forest",
    "a day at the seaside",
    "a friendly dragon who loves cookies",
    "a robot learning to paint",
    "a secret garden behind the school",
    "a flying bicycle",
    "a tree that can talk",
    "a brave little mouse",
    "a princess who wants to be a chef",
    "a rainbow that appears at night",
    "a tiny kingdom under a mushroom",
    "a mermaid who visits the shore",
    "a map that leads to a surprise",
    "a journey to the moon in a cardboard rocket",
    "a ghost who is afraid of the dark",
    "an elephant who forgets where she put her peanuts",
    "a penguin who wants to fly",
    "a snowman who comes to life at noon",
    "a dog who writes letters to his owner",
]

_OPENING_SENTENCES = [
    "Once upon a time there was a small village hidden between two mountains.",
    "The old clock in the tower struck midnight and something magical happened.",
    "A little girl named Emma found a golden key on her way to school.",
    "The animals in the forest were having their annual talent show.",
    "It was the rainiest day of the year and the river was rising fast.",
    "A mysterious package arrived at the door with no return address.",
    "The last leaf on the old oak tree finally let go of its branch.",
    "In the attic under a dusty blanket lay a book that glowed with light.",
    "The playground was empty except for one small child on the swings.",
    "A baker discovered that his bread could make people tell the truth.",
    "The lighthouse keeper saw something glowing at the bottom of the sea.",
    "Two friends found a map tucked inside an old library book.",
    "The school bus took a wrong turn and ended up in a fairy tale.",
    "A young inventor built a machine that could talk to animals.",
    "The first snow of winter fell on the little town by the lake.",
    "Grandma's recipe book contained one recipe written in disappearing ink.",
    "The garden gnome moved to a different spot every night when no one was looking.",
    "A boy discovered he could hear the thoughts of his pet hamster.",
    "The town fountain started bubbling with rainbow-colored water.",
    "An old pirate ship appeared in the harbor after a terrible storm.",
]

_STYLES = ["happy", "sad", "funny"]

_FORMATS = [
    "exactly three sentences",
    "exactly five sentences",
    "exactly two sentences",
    "no more than four sentences",
    "at least three but no more than five sentences",
    "exactly four sentences",
    "exactly three sentences",
    "exactly three sentences",
    "exactly three sentences",
    "exactly three sentences",
]


# ---------------------------------------------------------------------------
# Prompt generators
# ---------------------------------------------------------------------------

def _make_keyword_story(count: int = 40) -> list[dict]:
    """Type 1: keyword_story prompts."""
    prompts = []
    for i in range(count):
        words = _WORD_SETS[i % len(_WORD_SETS)]
        # Rotate which 3 words are used for variety beyond the bank
        if i >= len(_WORD_SETS):
            # Create new combinations by shuffling existing word sets
            base = _WORD_SETS[i % len(_WORD_SETS)][:]
            extra_idx = (i // len(_WORD_SETS)) % len(_WORD_SETS)
            swap_word = _WORD_SETS[extra_idx][i % 3]
            base[i % 3] = swap_word
            words = base
        prompts.append({
            "type": "keyword_story",
            "prompt": f"Write a short story using these words: {', '.join(words)}.",
            "required_words": words,
        })
    return prompts


def _make_topic_story(count: int = 20) -> list[dict]:
    """Type 2: topic_story prompts with emotion."""
    rng = random.Random(100)  # fixed seed for reproducibility
    prompts = []
    for i in range(count):
        topic = _TOPICS[i % len(_TOPICS)]
        emotion = _STYLES[i % len(_STYLES)]
        prompts.append({
            "type": "topic_story",
            "prompt": f"Write a {emotion} story about {topic}.",
            "required_words": [],
        })
    return prompts


def _make_continue_story(count: int = 20) -> list[dict]:
    """Type 3: continue_story prompts."""
    prompts = []
    for i in range(count):
        opening = _OPENING_SENTENCES[i % len(_OPENING_SENTENCES)]
        prompts.append({
            "type": "continue_story",
            "prompt": f"Continue the story: {opening}",
            "required_words": [],
        })
    return prompts


def _make_style_control(count: int = 10) -> list[dict]:
    """Type 4: style_control prompts."""
    prompts = []
    for i in range(count):
        topic = _TOPICS[(i + 5) % len(_TOPICS)]
        # Cycle through styles deterministically
        emotion = _STYLES[i % len(_STYLES)]
        prompts.append({
            "type": "style_control",
            "prompt": f"Write a {emotion} story about {topic}.",
            "required_words": [],
        })
    return prompts


def _make_format_control(count: int = 10) -> list[dict]:
    """Type 5: format_control prompts."""
    prompts = []
    for i in range(count):
        topic = _TOPICS[(i + 10) % len(_TOPICS)]
        fmt = _FORMATS[i % len(_FORMATS)]
        prompts.append({
            "type": "format_control",
            "prompt": f"Write a story in {fmt} about {topic}.",
            "required_words": [],
        })
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed evaluation prompts covering 5 types."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/prompts/eval_prompts.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Total number of prompts to generate (default: 100).",
    )
    args = parser.parse_args()

    # Calculate per-type allocation following the spec ratios
    # keyword: 40%, topic: 20%, continue: 20%, style: 10%, format: 10%
    n = args.num_prompts
    n_keyword = n * 40 // 100
    n_topic = n * 20 // 100
    n_continue = n * 20 // 100
    n_style = n * 10 // 100
    n_format = n - n_keyword - n_topic - n_continue - n_style  # remainder

    # Generate
    prompts = []
    prompts.extend(_make_keyword_story(n_keyword))
    prompts.extend(_make_topic_story(n_topic))
    prompts.extend(_make_continue_story(n_continue))
    prompts.extend(_make_style_control(n_style))
    prompts.extend(_make_format_control(n_format))

    print(f"Generated {len(prompts)} evaluation prompts:")
    type_counts = {}
    for p in prompts:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prompts)} prompts to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
