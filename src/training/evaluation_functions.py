from collections import defaultdict

# Evaluation uses the local metrics below; importing this module needs no HF downloads.

TRAILING_ANSWER_PUNCTUATION = " \t\r\n.,:;!?*_-`'\"<>|/\\()[]{}"


def split_sequence_at_special_tokens(sequence, special_token_sequence, SPECIAL_TOKENS):
    """
    Splits the sequence at the first occurrence of the special token sequence.

    Args:
        sequence (list): The input sequence to be split.
        special_token_sequence (list): The sequence of special tokens to split at.

    Returns:
        tuple: A tuple containing two lists - the part before the special token sequence
               and the part after it (including the special token sequence).
    """
    seq_len = len(sequence)
    token_seq_len = len(special_token_sequence)

    for i in range(seq_len - token_seq_len + 1):
        if sequence[i:i + token_seq_len] == special_token_sequence:
            ret = sequence[(i + token_seq_len):]
            ret = [tok for tok in ret if tok not in SPECIAL_TOKENS]
            return ret  

    return [tok for tok in sequence if tok not in SPECIAL_TOKENS]  # Return the whole sequence and an empty list if not found

def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax predictions before accumulation to avoid OOM.
    Without this, the trainer accumulates full (batch, seq_len, vocab_size) tensors
    across all eval batches, which can easily exceed available RAM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)



def _try_parse_number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize_prediction(text, options):
    """Remove categorical suffix fragments using options only, never gold.

    Accept punctuation or one alphabetic fragment separated by whitespace or
    starting with an uppercase letter. Preserve numbers and ambiguous answers.
    ``number`` is a placeholder, not a categorical answer to extract.
    """
    if options is None or text in options or _try_parse_number(text) is not None:
        return text
    categorical_options = {
        option for option in options
        if option and option != "number" and _try_parse_number(option) is None
    }

    def accepted_suffix(suffix):
        fragment = suffix.strip(TRAILING_ANSWER_PUNCTUATION)
        if fragment in {"", "The"}:
            return True
        # Do not discard a second answer, even when it is glued to the first.
        if fragment in categorical_options or fragment in {"Y", "N", "A", "P", "U", "V"}:
            return False
        if fragment.lower() in {"yes", "no", "unknown"}:
            return False
        if "/" in suffix or "|" in suffix:
            return False
        # Whitespace or capitalization marks a plausible extra word/token
        # fragment. Lowercase continuations such as Nega remain untouched.
        return fragment.isalpha() and (
            suffix[0].isspace() or fragment[0].isupper()
        )

    matches = [
        option for option in categorical_options
        if text.startswith(option)
        and accepted_suffix(text[len(option):])
    ]
    return matches[0] if len(matches) == 1 else text


def _compute_metrics(gold, pred, accepted_diff=0):
    """
    Compute accuracy, macro F1, micro F1, and weighted F1 for flat gold/pred lists.

    accepted_diff: percentage tolerance for numeric comparisons (0 = exact match).
    Returns: (accuracy, macro_f1, micro_f1, weighted_f1, per_class_f1)
    """
    if not gold:
        return 0.0, 0.0, 0.0, 0.0, {}

    def _distance(a, b):
        na, nb = _try_parse_number(a), _try_parse_number(b)
        if na is not None and nb is not None:
            return abs(na - nb) / abs(nb) if nb != 0 else abs(na)
        return 0.0 if a == b else float("inf")

    def _match(g, p):
        return _distance(g, p) <= accepted_diff / 100.0

    gold_classes = sorted(set(gold), key=lambda x: (_try_parse_number(x) is None, str(x)))
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in gold_classes}
    total_tp = total_fp = total_fn = 0

    for g, p in zip(gold, pred):
        per_class[g]["support"] += 1
        if _match(g, p):
            per_class[g]["tp"] += 1
            total_tp += 1
        else:
            per_class[g]["fn"] += 1
            total_fn += 1
            total_fp += 1
            if p in per_class:
                per_class[p]["fp"] += 1

    total_support = sum(per_class[c]["support"] for c in gold_classes)

    per_class_f1 = {}
    for c in gold_classes:
        tp = per_class[c]["tp"]
        fp = per_class[c]["fp"]
        fn = per_class[c]["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    macro_f1 = sum(per_class_f1.values()) / len(gold_classes) if gold_classes else 0.0

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    weighted_f1 = (
        sum(per_class_f1[c] * per_class[c]["support"] for c in gold_classes) / total_support
        if total_support > 0 else 0.0
    )

    accuracy = sum(1 for g, p in zip(gold, pred) if _match(g, p)) / len(gold)

    return accuracy, macro_f1, micro_f1, weighted_f1, per_class_f1


# ── public API ────────────────────────────────────────────────────────────────

def compute_metrics_crf_gamma(
    p,
    SPECIAL_TOKENS_ASSISTANT_START,
    SPECIAL_TOKENS,
    items_list_in_dataset_validation: list = [],
    items_list_in_dataset_test: list = [],
    calc_per_item: bool = True,
    verbose: bool = True,
    tokenizer=None,
    prediction_records: list | None = None,
    options_list_in_dataset_validation: list | None = None,
    options_list_in_dataset_test: list | None = None,
):
    """
    Compute evaluation metrics for the CRF extraction task during training.

    Existing metric keys score normalized answers; ``strict_`` keys score the
    original decoded answers. Options must align with their respective splits;
    when options are unavailable, predictions are left unchanged.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided for compute_metrics_crf_gamma.")
    predictions, labels = p
    # `label_names` includes both `labels` and `loss_weight_mask`, so the
    # Trainer returns a tuple of label arrays. Only the token labels belong in
    # generation metrics.
    labels = labels[0]

    if verbose:
        print("One label sequence:\n", labels[0])
        print("One prediction sequence:\n", predictions[0])

    # A decoder-only causal LM's logits at position t predict the target token
    # at t + 1. This must match the one-token shift used in
    # MedLlmTrainer.compute_loss(). Without it, an answer such as `N` is
    # compared to the following EOS token and decodes as an empty prediction.
    true_predictions = [
        [int(pred_tok) for pred_tok, lbl_tok in zip(prediction[:-1], label[1:]) if lbl_tok != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [int(lbl_tok) for pred_tok, lbl_tok in zip(prediction[:-1], label[1:]) if lbl_tok != -100]
        for prediction, label in zip(predictions, labels)
    ]

    def _find_subsequence(sequence, subsequence):
        subseq_len = len(subsequence)
        if subseq_len == 0:
            return None
        for i in range(len(sequence) - subseq_len + 1):
            if sequence[i:i + subseq_len] == subsequence:
                return i
        return None

    def _extract_response_from_label_span(prediction, label):
        start = _find_subsequence(label, SPECIAL_TOKENS_ASSISTANT_START)
        if start is None:
            label_response = split_sequence_at_special_tokens(
                label,
                SPECIAL_TOKENS_ASSISTANT_START,
                SPECIAL_TOKENS,
            )
            if not label_response:
                return [], []
            return prediction[-len(label_response):], label_response

        response_start = start + len(SPECIAL_TOKENS_ASSISTANT_START)
        label_response = label[response_start:]
        prediction_response = prediction[response_start:response_start + len(label_response)]
        return prediction_response, label_response

    # Isolate the response using the label-side span. Predicted argmax tokens
    # need not contain the assistant marker reliably, so never search for it
    # in predictions themselves.
    response_pairs = [
        _extract_response_from_label_span(tp, tl)
        for tp, tl in zip(true_predictions, true_labels)
    ]
    true_predictions = [pred for pred, label in response_pairs]
    true_labels = [label for pred, label in response_pairs]

    def _decode_or_join(token_sequences, empty_value):
        def _normalize_answer_text(text):
            re = __import__("re")
            text = str(text).strip()
            text = re.sub(r"(?is)<think\b[^>]*>.*?</think>", " ", text)
            text = re.sub(r"(?is)^.*?</think>", " ", text)
            text = re.sub(r"(?is)<think\b[^>]*>.*$", " ", text)
            text = re.sub(r"(?is)</?think\b[^>]*>", " ", text)
            # remove "thought"
            text = re.sub(r"(?is)<thought\b[^>]*>.*?</thought>", " ", text)
            text = re.sub(r"(?is)^.*?</thought>", " ", text)
            text = re.sub(r"(?is)<thought\b[^>]*>.*$", " ", text)
            text = re.sub(r"(?is)</?thought\b[^>]*>", " ", text)
            # remove "thought" when it is not a tag
            text = re.sub(r"(?is)\bthought\b", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        decoded = [
            _normalize_answer_text(tokenizer.decode(toks, skip_special_tokens=True)).replace("thought", "").strip()
            for toks in token_sequences
        ]
        return [text if text else empty_value for text in decoded]

    pred_str = _decode_or_join(true_predictions, "__invalid__")
    label_str = _decode_or_join(true_labels, "__missing__")

    if len(items_list_in_dataset_validation) == len(label_str):
        record_items = items_list_in_dataset_validation
        record_options = options_list_in_dataset_validation
    elif len(items_list_in_dataset_test) == len(label_str):
        record_items = items_list_in_dataset_test
        record_options = options_list_in_dataset_test
    else:
        record_items = None
        record_options = None
    if record_options is not None and len(record_options) != len(label_str):
        raise ValueError("Options must have one entry per evaluation example.")

    pred_str_raw = pred_str
    pred_str = [
        normalize_prediction(text, options)
        for text, options in zip(
            pred_str_raw,
            record_options if record_options is not None else [None] * len(pred_str_raw),
        )
    ]

    # Optionally expose the decoded examples to evaluation entrypoints that
    # need to persist predictions in addition to aggregate metrics. Mutating a
    # caller-owned list keeps the Trainer-compatible return value unchanged.
    if prediction_records is not None:
        prediction_records.clear()
        prediction_records.extend(
            {
                "index": index,
                "crf_item": item,
                "gold": gold,
                "prediction": prediction,
                "prediction_raw": raw,
                "options": options,
            }
            for index, (item, gold, prediction, raw, options) in enumerate(
                zip(
                    record_items if record_items is not None else [None] * len(label_str),
                    label_str,
                    pred_str,
                    pred_str_raw,
                    record_options if record_options is not None else [None] * len(label_str),
                )
            )
        )

    if verbose:
        print("One prediction (str):", pred_str[0])
        print("One label (str):", label_str[0])

    if calc_per_item and record_items is None:
        print(
            "Warning: Length of items list does not match evaluation examples. "
            "Skipping per-item F1 calculation."
        )
    metrics = _prediction_metrics(label_str, pred_str, record_items, calc_per_item)
    strict_metrics = _prediction_metrics(label_str, pred_str_raw, record_items, calc_per_item)
    metrics.update({f"strict_{key}": value for key, value in strict_metrics.items()})
    metrics["AAA_postprocessed_predictions"] = sum(
        raw != normalized for raw, normalized in zip(pred_str_raw, pred_str)
    )
    return metrics


def _prediction_metrics(label_str, pred_str, items=None, calc_per_item=True):
    """Score either raw or normalized strings with the same metric definitions."""
    # Global metrics
    accuracy, macro_f1, micro_f1, weighted_f1, per_class_f1 = _compute_metrics(label_str, pred_str)

    # only classification metrics (exclude numbers)
    pred_str_class, label_str_class = [], []
    pred_str_num, label_str_num = [], []
    def try_parse_number(x):
        try:
            return float(x)
        except:
            return None
        
    for p, l in zip(pred_str, label_str):
        if try_parse_number(l) is not None:
            pred_str_num.append(p)
            label_str_num.append(l)
        else:
            pred_str_class.append(p)
            label_str_class.append(l)
    
    accuracy_class, macro_f1_class, micro_f1_class, weighted_f1_class, per_class_f1_class = _compute_metrics(label_str_class, pred_str_class)
    accuracy_num, _, _, _, _ = _compute_metrics(label_str_num, pred_str_num)

    # Per-item metrics
    f1_scores_per_item = None
    if calc_per_item and items is not None:
        gold_per_item: dict[str, list] = defaultdict(list)
        pred_per_item: dict[str, list] = defaultdict(list)
        for item, g, p in zip(items, label_str, pred_str):
            gold_per_item[item].append(g)
            pred_per_item[item].append(p)

        f1_scores_per_item = {}
        for item in gold_per_item:
            _, item_macro, _, _, _ = _compute_metrics(gold_per_item[item], pred_per_item[item])
            f1_scores_per_item[item] = item_macro

    return {
        "AAA_accuracy": accuracy,
        "AAA_f1_micro": micro_f1,
        "AAA_f1_macro": macro_f1,
        "AAA_f1_weighted": weighted_f1,
        "AAA_accuracy_categorical": accuracy_class,
        "AAA_f1_micro_categorical": micro_f1_class,
        "AAA_f1_macro_categorical": macro_f1_class,
        "AAA_f1_weighted_categorical": weighted_f1_class,
        "AAA_accuracy_numerical": accuracy_num,
        "class/f1_results_per_class": per_class_f1,
        "items/f1_scores_per_item": f1_scores_per_item,
    }
