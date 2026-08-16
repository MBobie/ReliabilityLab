"""Text perturbations for robustness testing."""

import random
import string


def typo_swap(text: str, rng: random.Random) -> str:
    """Introduce one adjacent-character swap in an eligible word."""

    words = text.split()

    eligible = [
        i for i, word in enumerate(words)
        if len(word.strip(string.punctuation)) >= 4
    ]

    if not eligible:
        return text

    word_index = rng.choice(eligible)

    word = words[word_index]

    # Avoid punctuation at the edges where possible
    candidate_positions = [
        i
        for i in range(len(word) - 1)
        if word[i].isalpha() and word[i + 1].isalpha()
    ]

    if not candidate_positions:
        return text

    position = rng.choice(candidate_positions)

    chars = list(word)

    chars[position], chars[position + 1] = (
        chars[position + 1],
        chars[position],
    )

    words[word_index] = "".join(chars)

    return " ".join(words)


def character_deletion(
    text: str,
    rng: random.Random,
) -> str:
    """Delete one alphabetic character from an eligible word."""

    words = text.split()

    eligible = [
        i for i, word in enumerate(words)
        if sum(char.isalpha() for char in word) >= 4
    ]

    if not eligible:
        return text

    word_index = rng.choice(eligible)

    word = words[word_index]

    candidate_positions = [
        i
        for i, char in enumerate(word)
        if char.isalpha()
    ]

    if not candidate_positions:
        return text

    position = rng.choice(candidate_positions)

    words[word_index] = (
        word[:position]
        + word[position + 1:]
    )

    return " ".join(words)


def word_deletion(
    text: str,
    rng: random.Random,
) -> str:
    """Delete one word from the input text."""

    words = text.split()

    if len(words) <= 1:
        return text

    index = rng.randrange(len(words))

    del words[index]

    return " ".join(words)


def case_change(text: str) -> str:
    """Convert text to uppercase."""

    return text.upper()


def punctuation_noise(
    text: str,
    rng: random.Random,
) -> str:
    """Inject extra punctuation into the text."""

    punctuation = [
        "!!!",
        "???",
        "...",
        "?!",
    ]

    noise = rng.choice(punctuation)

    words = text.split()

    if not words:
        return text

    index = rng.randrange(len(words))

    words[index] = words[index] + noise

    return " ".join(words)


def perturb_texts(
    texts,
    perturbation: str,
    seed: int = 42,
):
    """Apply a selected perturbation to a sequence of texts."""

    rng = random.Random(seed)

    outputs = []

    for text in texts:

        if perturbation == "typo":
            transformed = typo_swap(text, rng)

        elif perturbation == "char_delete":
            transformed = character_deletion(
                text,
                rng,
            )

        elif perturbation == "word_delete":
            transformed = word_deletion(
                text,
                rng,
            )

        elif perturbation == "case":
            transformed = case_change(text)

        elif perturbation == "punctuation":
            transformed = punctuation_noise(
                text,
                rng,
            )

        else:
            raise ValueError(
                f"Unknown perturbation: {perturbation}"
            )

        outputs.append(transformed)

    return outputs

def _select_count(total: int, severity: float) -> int:
    """Convert a severity fraction into a number of items to perturb."""

    if not 0 < severity <= 1:
        raise ValueError(
            "severity must be greater than 0 and at most 1."
        )

    if total == 0:
        return 0

    return max(
        1,
        round(total * severity),
    )


def typo_swap_severity(
    text: str,
    rng: random.Random,
    severity: float,
) -> str:
    """Swap adjacent characters in a fraction of eligible words."""

    words = text.split()

    eligible = []

    for word_index, word in enumerate(words):

        candidate_positions = [
            i
            for i in range(len(word) - 1)
            if word[i].isalpha()
            and word[i + 1].isalpha()
        ]

        if candidate_positions:
            eligible.append(
                (
                    word_index,
                    candidate_positions,
                )
            )

    count = _select_count(
        len(eligible),
        severity,
    )

    if count == 0:
        return text

    selected = rng.sample(
        eligible,
        k=min(count, len(eligible)),
    )

    for word_index, candidate_positions in selected:

        word = words[word_index]

        position = rng.choice(
            candidate_positions
        )

        chars = list(word)

        chars[position], chars[position + 1] = (
            chars[position + 1],
            chars[position],
        )

        words[word_index] = "".join(chars)

    return " ".join(words)


def character_deletion_severity(
    text: str,
    rng: random.Random,
    severity: float,
) -> str:
    """Delete one character from a fraction of eligible words."""

    words = text.split()

    eligible = []

    for word_index, word in enumerate(words):

        candidate_positions = [
            i
            for i, char in enumerate(word)
            if char.isalpha()
        ]

        if len(candidate_positions) >= 4:
            eligible.append(
                (
                    word_index,
                    candidate_positions,
                )
            )

    count = _select_count(
        len(eligible),
        severity,
    )

    if count == 0:
        return text

    selected = rng.sample(
        eligible,
        k=min(count, len(eligible)),
    )

    for word_index, candidate_positions in selected:

        word = words[word_index]

        position = rng.choice(
            candidate_positions
        )

        words[word_index] = (
            word[:position]
            + word[position + 1:]
        )

    return " ".join(words)


def word_deletion_severity(
    text: str,
    rng: random.Random,
    severity: float,
) -> str:
    """Delete a fraction of words from the input."""

    words = text.split()

    if len(words) <= 1:
        return text

    count = _select_count(
        len(words),
        severity,
    )

    count = min(
        count,
        len(words) - 1,
    )

    indices = set(
        rng.sample(
            range(len(words)),
            k=count,
        )
    )

    remaining = [
        word
        for index, word in enumerate(words)
        if index not in indices
    ]

    return " ".join(remaining)


def perturb_texts_with_severity(
    texts,
    perturbation: str,
    severity: float,
    seed: int = 42,
):
    """Apply a perturbation at a specified severity level."""

    rng = random.Random(seed)

    outputs = []

    for text in texts:

        if perturbation == "typo":

            transformed = typo_swap_severity(
                text,
                rng,
                severity,
            )

        elif perturbation == "char_delete":

            transformed = character_deletion_severity(
                text,
                rng,
                severity,
            )

        elif perturbation == "word_delete":

            transformed = word_deletion_severity(
                text,
                rng,
                severity,
            )

        else:
            raise ValueError(
                f"Unknown severity perturbation: "
                f"{perturbation}"
            )

        outputs.append(transformed)

    return outputs

def perturb_texts_with_severity(
    texts,
    perturbation: str,
    severity: float,
    seed: int = 42,
):
    """Apply a perturbation at a specified severity level."""

    rng = random.Random(seed)

    outputs = []

    for text in texts:

        if perturbation == "typo":
            transformed = typo_swap_severity(
                text,
                rng,
                severity,
            )

        elif perturbation == "char_delete":
            transformed = character_deletion_severity(
                text,
                rng,
                severity,
            )

        elif perturbation == "word_delete":
            transformed = word_deletion_severity(
                text,
                rng,
                severity,
            )

        else:
            raise ValueError(
                f"Unknown severity perturbation: {perturbation}"
            )

        outputs.append(transformed)

    return outputs  

def perturb_texts_probabilistic(
    texts,
    perturbation: str,
    severity: float,
    seed: int = 42,
    return_stats: bool = False,
):
    """Apply probabilistic text corruption.

    Each eligible unit is independently perturbed with probability
    equal to ``severity``.

    Parameters
    ----------
    texts
        Sequence of input strings.
    perturbation
        One of "typo", "char_delete", or "word_delete".
    severity
        Probability of perturbing an eligible unit, between 0 and 1.
    seed
        Random seed controlling perturbation realization.
    return_stats
        If True, also return corruption statistics.

    Returns
    -------
    list
        Perturbed texts.

    tuple
        If return_stats=True, returns (texts, statistics).
    """

    if not 0.0 <= severity <= 1.0:
        raise ValueError(
            "severity must be between 0 and 1."
        )

    rng = random.Random(seed)

    outputs = []

    eligible_total = 0
    affected_total = 0

    for text in texts:

        words = text.split()

        # -----------------------------------------------------
        # TYPO
        # -----------------------------------------------------
        if perturbation == "typo":

            for word_index, word in enumerate(words):

                candidate_positions = [
                    i
                    for i in range(len(word) - 1)
                    if word[i].isalpha()
                    and word[i + 1].isalpha()
                ]

                if not candidate_positions:
                    continue

                eligible_total += 1

                if rng.random() < severity:

                    position = rng.choice(
                        candidate_positions
                    )

                    chars = list(word)

                    chars[position], chars[position + 1] = (
                        chars[position + 1],
                        chars[position],
                    )

                    words[word_index] = "".join(chars)

                    affected_total += 1

        # -----------------------------------------------------
        # CHARACTER DELETION
        # -----------------------------------------------------
        elif perturbation == "char_delete":

            for word_index, word in enumerate(words):

                candidate_positions = [
                    i
                    for i, char in enumerate(word)
                    if char.isalpha()
                ]

                if len(candidate_positions) < 4:
                    continue

                eligible_total += 1

                if rng.random() < severity:

                    position = rng.choice(
                        candidate_positions
                    )

                    words[word_index] = (
                        word[:position]
                        + word[position + 1:]
                    )

                    affected_total += 1

        # -----------------------------------------------------
        # WORD DELETION
        # -----------------------------------------------------
        elif perturbation == "word_delete":

            if len(words) <= 1:
                outputs.append(text)
                continue

            eligible_total += len(words)

            deletion_indices = []

            for word_index in range(len(words)):

                if rng.random() < severity:
                    deletion_indices.append(
                        word_index
                    )

            # Never allow an utterance to become completely empty.
            if len(deletion_indices) == len(words):

                keep_index = rng.choice(
                    deletion_indices
                )

                deletion_indices.remove(
                    keep_index
                )

            deletion_set = set(
                deletion_indices
            )

            affected_total += len(
                deletion_indices
            )

            words = [
                word
                for index, word in enumerate(words)
                if index not in deletion_set
            ]

        else:
            raise ValueError(
                f"Unknown probabilistic perturbation: "
                f"{perturbation}"
            )

        outputs.append(
            " ".join(words)
        )

    realized_severity = (
        affected_total / eligible_total
        if eligible_total > 0
        else 0.0
    )

    stats = {
        "requested_severity": severity,
        "eligible_units": eligible_total,
        "affected_units": affected_total,
        "realized_severity": realized_severity,
    }

    if return_stats:
        return outputs, stats

    return outputs    