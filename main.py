import argparse
import json
import os
from typing import Dict, Any, Tuple

# Load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # looks for a .env file in the current working directory or parents
except Exception:
    pass

from audio_rating import AudioCriteriaRater
from cry_detector import CryDetector
from model_prediction import predict_distribution_from_rater_scores


def _call_openai_for_adjustments(
    model_name: str,
    probs: Dict[str, float],
    rater_scores: list[tuple[str, float]],
    user_context: str,
) -> Tuple[Dict[str, float], str, Dict[str, float], str]:
    """Ask an LLM to propose small (±10%) adjustments.

    Returns (adjusted_probs, reasoning, raw_adjustments_percent)
    On failure, returns original probs and a default reasoning.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM ERROR] No OPENAI_API_KEY found in environment.")
        return probs, "No LLM changes applied (missing OPENAI_API_KEY).", {k: 0.0 for k in probs}, "Based on the cry's qualities and your context, results are unchanged."

    # Prepare a compact, deterministic input
    ordered_classes = list(probs.keys())
    criteria_payload = [{"criterion": c, "score": float(s)} for c, s in rater_scores]
    sorted_probs = sorted(((k, float(v)) for k, v in probs.items()), key=lambda kv: kv[1], reverse=True)
    top_class = sorted_probs[0][0] if sorted_probs else ""

    system_prompt = (
        "You are a careful assistant helping refine probabilities over 5 baby-cry classes. "
        "You may ONLY make small adjustments up to ±10% for each class. "
        "Base changes on the provided criteria scores (1–10 scale) and user context. "
        "Return strictly valid JSON."
    )
    user_prompt = {
        "base_probabilities": {k: float(probs[k]) for k in ordered_classes},
        "sorted_probabilities": [{"class": k, "prob": v} for k, v in sorted_probs],
        "top_class": top_class,
        "criteria_scores": criteria_payload,
        "user_context": user_context or "",
        "instructions": (
            "Propose small adjustments to base_probabilities. For EACH class, return a percentage in range [-10, 10]. "
            "Keep the sum unconstrained; the caller will renormalize. Also provide: "
            "(1) a one-paragraph 'reasoning' under 200 words for developers, and "
            "(2) a one-paragraph 'user_summary' under 220 words in plain language explaining why the probabilities make sense, focusing on the top two classes."
            "Focus mostly on the top class, finding justification for it however you can. Do not talk about any class with less than 10% probability."
            "Reference cry characteristics and the provided context, "
            "without mentioning models, lists of criteria, or any changes/adjustments to probabilities. Do NOT imply that values were altered. "
            "Avoid words like 'adjusted', 'updated', 'refined', 'tweaked', 'now', 'original', or 'baseline'. Focus only on cry characteristics and the provided context. "
            "Output JSON with keys: 'adjustments' (object of class->percent), 'reasoning' (string), 'user_summary' (string)."
        ),
        "class_order": ordered_classes,
    }

    content: str | None = None
    # Try modern OpenAI client first
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content  # type: ignore[attr-defined]
    except Exception as e1:
        print(f"[LLM ERROR] Modern OpenAI client failed: {e1}")
        # Fallback to legacy openai lib if available
        try:
            import openai  # type: ignore

            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception as e2:
            print(f"[LLM ERROR] Legacy OpenAI client failed: {e2}")
            content = None

    if not content:
        print("[LLM ERROR] No content returned from LLM call.")
        return probs, "No LLM changes applied (LLM call failed).", {k: 0.0 for k in probs}, "Based on the cry's qualities and your context, results are unchanged."

    # Parse JSON from the model output
    try:
        # Extract JSON if the assistant wrapped it in text
        content_str = content.strip()
        json_start = content_str.find("{")
        json_end = content_str.rfind("}")
        if json_start != -1 and json_end != -1:
            content_str = content_str[json_start : json_end + 1]
        payload = json.loads(content_str)
    except Exception as e3:
        print(f"[LLM ERROR] Failed to parse JSON from LLM output: {e3}")
        print(f"[LLM ERROR] Raw LLM output: {content}")
        return probs, "No LLM changes applied (invalid JSON).", {k: 0.0 for k in probs}, "Based on the cry's qualities and your context, results are unchanged."

    raw_adj = payload.get("adjustments", {})
    reasoning = payload.get("reasoning", "")
    user_summary = payload.get("user_summary", "")

    # Build percentage adjustments for all classes with clamping to [-10, 10]
    percents: Dict[str, float] = {}
    for cls in ordered_classes:
        val = float(raw_adj.get(cls, 0.0))
        if val > 10.0:
            val = 10.0
        elif val < -10.0:
            val = -10.0
        percents[cls] = val

    # Apply adjustments multiplicatively and renormalize
    adjusted = {}
    for cls in ordered_classes:
        factor = 1.0 + (percents[cls] / 100.0)
        adjusted[cls] = max(0.0, probs[cls] * factor)

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    # Ensure numerical sanity
    adjusted = {k: float(v) for k, v in adjusted.items()}
    reasoning_text = reasoning.strip() or "Minor refinements not applied."
    user_summary_text = user_summary.strip() or "Based on what we hear in the cry and your context, these probabilities reflect the most likely needs right now."
    return adjusted, reasoning_text, percents, user_summary_text


def _call_openai_for_user_summary(
    model_name: str,
    final_probs: Dict[str, float],
    rater_scores: list[tuple[str, float]],
    user_context: str,
) -> str:
    """Ask an LLM to produce a user-facing summary based on FINAL probabilities.

    Returns the summary string. On failure, returns a safe default message.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Based on what we hear and your context, these probabilities reflect the most likely needs right now."

    ordered_classes = list(final_probs.keys())
    criteria_payload = [{"criterion": c, "score": float(s)} for c, s in rater_scores]
    sorted_probs = sorted(((k, float(v)) for k, v in final_probs.items()), key=lambda kv: kv[1], reverse=True)
    top_class = sorted_probs[0][0] if sorted_probs else ""

    system_prompt = (
        "You are generating a concise, user-friendly explanation for why the probabilities make sense. "
        "Do NOT mention models, criteria lists, probabilities being adjusted/changed, or any internal process. "
        "Use only the provided class names. Focus on cry characteristics and the provided context."
    )

    user_prompt = {
        "final_probabilities": {k: float(final_probs[k]) for k in ordered_classes},
        "sorted_probabilities": [{"class": k, "prob": v} for k, v in sorted_probs],
        "top_class": top_class,
        "criteria_scores": criteria_payload,
        "user_context": user_context or "",
        "instructions": (
            "Write one paragraph under 80 words as 'user_summary' in JSON. "
            "Do not imply any change in values. Avoid words like 'adjusted', 'updated', 'refined', 'tweaked', 'now', 'original', 'baseline'. "
            "When referencing likely needs, prioritize 'top_class' and optionally the next one. Use only provided class names. "
            "Output JSON with key 'user_summary' only."
        ),
        "class_order": ordered_classes,
    }

    content: str | None = None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        try:
            import openai  # type: ignore

            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                temperature=0.2,
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            content = None

    if not content:
        return "Based on what we hear and your context, these probabilities reflect the most likely needs right now."

    try:
        content_str = content.strip()
        json_start = content_str.find("{")
        json_end = content_str.rfind("}")
        if json_start != -1 and json_end != -1:
            content_str = content_str[json_start : json_end + 1]
        payload = json.loads(content_str)
        summary = str(payload.get("user_summary", "")).strip()
        if not summary:
            raise ValueError("empty summary")
        return summary
    except Exception:
        return "Based on what we hear and your context, these probabilities reflect the most likely needs right now."


def run_inference(audio_path: str, criteria_file: str, model_path: str, threshold: float, skip_cry_check: bool, user_context: str, openai_model: str, disable_llm: bool) -> Dict[str, Any]:
    # Optional cry detection gate
    cry_info = None
    if not skip_cry_check:
        detector = CryDetector()
        cry_info = detector.detect_cry(audio_path, threshold=threshold)
        if not cry_info["is_cry"]:
            return {
                "is_cry": False,
                "cry_confidence": cry_info["cry_confidence"],
                "probabilities": {},
            }

    # Compute similarity scores for all criteria using the CLAP rater
    rater = AudioCriteriaRater()
    with open(criteria_file, "r") as f:
        criteria = json.load(f).get("criteria", [])

    rater_scores = rater.compute_similarity(audio_path, criteria)

    # Predict probabilities with calibrated model and average with prior
    probs = predict_distribution_from_rater_scores(
        rater_scores=rater_scores,
        model_path=model_path,
        criteria_path=criteria_file,
    )

    # LLM refinement step
    final_probs = probs
    reasoning = ""
    user_friendly_summary = ""
    adjustments = {k: 0.0 for k in probs}
    if not disable_llm:
        # First, get adjustments and developer-oriented reasoning
        final_probs, reasoning, adjustments, _ = _call_openai_for_adjustments(
            model_name=openai_model,
            probs=probs,
            rater_scores=rater_scores,
            user_context=user_context,
        )
        # Then, generate user-facing summary based on FINAL probabilities
        user_friendly_summary = _call_openai_for_user_summary(
            model_name=openai_model,
            final_probs=final_probs,
            rater_scores=rater_scores,
            user_context=user_context,
        )

    result: Dict[str, Any] = {
        "is_cry": True if skip_cry_check else cry_info["is_cry"],
        "cry_confidence": None if skip_cry_check else cry_info["cry_confidence"],
        "model_probabilities": probs,
        "final_probabilities": final_probs,
        "probabilities": final_probs,
        "llm_summary": reasoning,
        "llm_adjustments_percent": adjustments,
        "user_context": user_context or "",
        "llm_user_summary": user_friendly_summary,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict baby cry need probabilities from audio.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--criteria-file", type=str, default="default_criteria.json", help="JSON file with criteria list")
    parser.add_argument("--model-path", type=str, default="models/calibrated_clf.joblib", help="Path to calibrated classifier joblib")
    parser.add_argument("--threshold", type=float, default=0.1, help="Cry detection threshold (0-1)")
    parser.add_argument("--skip-cry-check", action="store_true", help="Skip cry detection gate and always run prediction")
    parser.add_argument("--user-context", type=str, default="", help="Additional context (e.g., 'baby just ate', 'teething today')")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini", help="OpenAI model name for LLM refinement")
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM refinement even if configured")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    result = run_inference(
        audio_path=args.audio_path,
        criteria_file=args.criteria_file,
        model_path=args.model_path,
        threshold=args.threshold,
        skip_cry_check=args.skip_cry_check,
        user_context=args.user_context,
        openai_model=args.openai_model,
        disable_llm=args.disable_llm,
    )

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()

