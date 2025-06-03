import re
import string
from collections import Counter
from typing import Dict, Tuple, Optional


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    special_answers = ["yes", "no", "no answer"]

    if normalized_prediction in special_answers or normalized_ground_truth in special_answers:
        if normalized_prediction in normalized_ground_truth.split() or normalized_ground_truth in normalized_prediction.split():
            return 1.0, 1.0, 1.0
        else:
            return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth))


def cover_exact_match_score_1(prediction, ground_truth):
    # 不考虑顺序和连续
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    return all(token in pre_list for token in ground_list)


def cover_exact_match_score_2(prediction, ground_truth):
    # 考虑顺序和连续
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()

    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True

    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)

    if ground_str in pre_str:
        return True

    return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            em_score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(em_score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, precision, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, precision, recall))
        f1, precision, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, precision, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    else:
        raise NotImplementedError


def compute_metrics(prediction, gold):
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, precision, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(cover_exact_match_score_1, prediction, gold)
    cover_em_2 = metric_max_over_ground_truths(cover_exact_match_score_2, prediction, gold)

    metrics = dict()
    metrics["em"] = float(em)
    metrics["cover_em_1"] = float(cover_em_1)
    metrics["cover_em_2"] = float(cover_em_2)
    metrics["f1"] = f1
    metrics["precision"] = precision
    metrics["recall"] = recall

    if cover_em_1:
        metrics["acc_num"] = 1

    return metrics


def validate_model_answer(answer_text: str, expected_answer: str):
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_answer: Text extracted from data

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    print(f"\n[Answer Validation]")

    print(f"  Expected answer: {expected_answer}")
    print(f"  Predicted answer: {answer_text}")

    if isinstance(expected_answer, list):
        metrics = compute_metrics(answer_text, expected_answer)
    else:
        metrics = compute_metrics(answer_text, [expected_answer])

    return metrics


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("  [Error] Failed to locate model response header")
        return None, solution_str

    # Extract reasoning and final answer using XML-style tags
    reasoning_pattern = r'<think>(.*?)</think>'
    matches = list(re.finditer(reasoning_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid reasoning text found")

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid answer text found")
        answer_text = None
    else:
        answer_text = matches[-1].group(1).strip()

    return answer_text, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Format Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False

    return validation_passed


def compute_score(solution_str: str,
                  ground_truth: str):
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data

    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))

    # Parse ground truth data
    print(f"\n[Ground Truth]  {ground_truth}")

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = 0 if format_correct else -2
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    # Validate answer content
    if format_correct and answer_text:
        metrics = validate_model_answer(answer_text, ground_truth)
        answer_score = metrics["f1"]
        print(f"  Answer score: {answer_score}")
    else:
        answer_score = 0
        error = "missing answer" if format_correct else "format errors"
        print(f"\n[Answer Validation] Skipped due to {error} (Answer score: {answer_score})")

    return format_score, answer_score
