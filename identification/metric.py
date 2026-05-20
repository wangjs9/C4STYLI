import argparse
import json
import os
import re
import sys

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_metrics(golden, predicted):
    valid_indices = [
        i
        for i, p in enumerate(predicted)
        if p in ["CN", "HK"] and golden[i] in ["CN", "HK"]
    ]
    valid_golden = [golden[i] for i in valid_indices]
    valid_predicted = [predicted[i] for i in valid_indices]

    acc = accuracy_score(valid_golden, valid_predicted)
    f1 = f1_score(valid_golden, valid_predicted, average="macro")
    precision = precision_score(
        valid_golden, valid_predicted, average="macro", zero_division=0
    )
    recall = recall_score(
        valid_golden, valid_predicted, average="macro", zero_division=0
    )

    print(f"ACC: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")

    metrics = {
        "acc": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "total_samples": len(golden),
        "valid_samples": len(valid_golden),
    }
    return metrics


def text_to_json(json_output):
    if "```json" not in json_output and "{" not in json_output:
        raise ValueError(f"Invalid json output: {json_output}")
    json_output = json_output.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(json_output)
    except json.JSONDecodeError as e:
        # Try to fix unescaped quotes inside string values using state machine
        fixed = json_output
        result = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(fixed):
            char = fixed[i]

            if escape_next:
                result.append(char)
                escape_next = False
            elif char == "\\":
                result.append(char)
                escape_next = True
            elif char == '"':
                if in_string:
                    # We're inside a string value
                    # Check what comes after this quote to determine if it closes the string
                    j = i + 1
                    # Skip whitespace
                    while j < len(fixed) and fixed[j] in " \n\t\r":
                        j += 1

                    if j >= len(fixed):
                        # End of input, this closes the string
                        result.append(char)
                        in_string = False
                    elif fixed[j] in ",}":
                        # Followed by comma or closing brace, this closes the string
                        result.append(char)
                        in_string = False
                    elif fixed[j] == '"':
                        # Consecutive quotes ""
                        # Check further: if followed by key pattern (":) or end pattern (,}),
                        # then first quote closes string
                        k = j + 1
                        while k < len(fixed) and fixed[k] in " \n\t\r":
                            k += 1
                        if k < len(fixed) and (fixed[k] == ":" or fixed[k] in ",}"):
                            # Pattern: ""key": or ""} or "", - first quote closes string
                            result.append(char)
                            in_string = False
                            # Skip the second quote in next iteration
                            i += 1
                            continue
                        else:
                            # Both quotes are inside the string value
                            # Check if second quote should also be escaped
                            k = j + 1
                            while k < len(fixed) and fixed[k] in " \n\t\r":
                                k += 1
                            if k < len(fixed) and fixed[k] not in ",}:":
                                # Second quote is also inside string, escape both quotes
                                result.append('\\"')
                                result.append('\\"')
                                i += 1  # Skip second quote
                                continue
                            else:
                                # Second quote closes the string, escape only the first
                                result.append('\\"')
                    elif fixed[j] == ":":
                        # This shouldn't happen in a value string, but handle it
                        result.append(char)
                        in_string = False
                    else:
                        # This is an unescaped quote inside the string value
                        result.append('\\"')
                else:
                    # We're not in a string, this starts a new string
                    # Check if it's a key (followed by :) or a value
                    j = i + 1
                    while j < len(fixed) and fixed[j] in " \n\t\r":
                        j += 1
                    if j < len(fixed) and fixed[j] == ":":
                        # This is a key name
                        result.append(char)
                        in_string = True
                    else:
                        # This is a value string
                        result.append(char)
                        in_string = True
            else:
                result.append(char)
            i += 1

        fixed = "".join(result)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # If still fails, raise with helpful error message
            raise ValueError(
                f"Invalid json output (could not fix): {json_output[:200]}..."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="identification/output/DeepSeekV3.2_slogans.json",
    )
    parser.add_argument(
        "--text_domain", type=str, default="slogans", choices=["titles", "slogans"]
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="identification/output/DeepSeekV3.2_slogans_metric.json",
    )
    args = parser.parse_args()

    data = json.load(open(args.dataset_path, "r", encoding="utf-8"))
    if args.text_domain == "titles":
        data = [item for item in data if item["domain"] == "titles"]
    elif args.text_domain == "slogans":
        data = [item for item in data if item["domain"] == "slogans"]
    else:
        raise ValueError(f"Invalid text domain: {args.text_domain}")

    golden = []
    predicted = []
    for idx, item in enumerate(data):
        output_text = item["output_text"]
        region = item["region"]
        json_output = re.sub(
            r"<think>.*?</think>", "", output_text, flags=re.DOTALL
        ).strip()
        json_data = text_to_json(json_output)
        is_mainland = json_data["is_mainland"]
        is_hongkong = json_data["is_hongkong"]
        golden.append(region)
        if is_mainland and is_hongkong:
            if region == "CN":
                predicted.append("HK")
            elif region == "HK":
                predicted.append("CN")
        elif is_mainland:
            predicted.append("CN")
        elif is_hongkong:
            predicted.append("HK")
        else:
            if region == "CN":
                predicted.append("HK")
            elif region == "HK":
                predicted.append("CN")

    metrics = calculate_metrics(golden, predicted)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    main()
