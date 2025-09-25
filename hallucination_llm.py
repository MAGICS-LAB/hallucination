import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from difflib import SequenceMatcher
import argparse


def simple_eval(pred, ref_list):
    """Compute max string similarity score against reference answers (higher is better)."""
    pred = pred.lower()
    scores = [SequenceMatcher(None, pred, ref.lower()).ratio() for ref in ref_list]
    return max(scores) if scores else 0.0


def common_wrong_eval(pred, common_wrong_list):
    """
    Compute max similarity to common incorrect answers (higher is worse).
    If empty, returns 0.
    """
    pred = pred.lower()
    scores = [SequenceMatcher(None, pred, wrong.lower()).ratio() for wrong in common_wrong_list]
    return max(scores) if scores else 0.0


def main(use_lora=True, num_samples=30):
    import pandas as pd

    # === 1. Load TruthfulQA dataset from CSV ===
    csv_path = "root_dir/TruthfulQA.csv"  # your path
    df = pd.read_csv(csv_path)
    
    val_set = []
    for _, row in df.iterrows():
        val_set.append({
            "question": row["Question"],
            "best_answer": row["Best Answer"],
            "correct_answers": [a.strip() for a in str(row["Correct Answers"]).split(";") if a.strip()],
            "common_wrong_answers": [a.strip() for a in str(row["Incorrect Answers"]).split(";") if a.strip()]
        })
    
    
    num_samples = len(val_set)


    # === 2. Load base model and tokenizer ===
    base_model_name = "root_dir/Qwen2.5-7B-Instruct"  # path or HF model name
    finetuned_path = "root_dir/Qwen2.5_7B_4epochs_2"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # === 3. Optionally load LoRA adapter ===
    model = base_model
    if use_lora:
        model = PeftModel.from_pretrained(base_model, finetuned_path)
    model.eval()

    # === 4. Prepare common wrong answers (dummy example) ===
    common_wrongs_all = [val_set[i].get("common_wrong_answers", []) for i in range(len(val_set))]

    total_score_correct = 0.0
    total_score_wrong = 0.0


    num_samples = min(num_samples, len(val_set))

    for i in range(num_samples):
        q = val_set[i]["question"]
        best = val_set[i]["best_answer"]
        common_wrong = common_wrongs_all[i]


        inputs = tokenizer(q, return_tensors="pt").to(model.device)


        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)


        score_wrong = common_wrong_eval(pred, common_wrong)

        total_score_wrong += score_wrong


        print("=" * 50)
        print(f"Q{i+1}: {q}")
        print(f"Model: {pred}")
        print(f"Best Answer: {best}")
        print(f"Common wrong similarity: {score_wrong:.3f}")

    avg_score_wrong = total_score_wrong / num_samples
    print("Average similarity to common wrong answers:", avg_score_wrong)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapter instead of base model")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of validation samples to evaluate")
    args = parser.parse_args()

    main(use_lora=args.use_lora, num_samples=args.num_samples)