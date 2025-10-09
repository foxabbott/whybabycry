import json
from typing import Any, Dict, List, Tuple

import gradio as gr

from main import run_inference


def _format_probabilities_table(probabilities: Dict[str, float]) -> List[List[Any]]:
    items: List[Tuple[str, float]] = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    rows: List[List[Any]] = []
    for label, prob in items:
        percent = int(round(float(prob) * 100))
        display_label = label.replace("_", " ")
        rows.append([display_label, f"{percent}%"])
    return rows


def analyze(audio_path: str, user_context: str) -> Tuple[str, Any, Any, Any, str, Any]:
    if not audio_path:
        return (
            "Please upload or record some audio.",
            gr.update(value=[], visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            "",
            gr.update(visible=False),
        )

    result = run_inference(
        audio_path=audio_path,
        criteria_file="default_criteria.json",
        model_path="models/calibrated_clf.joblib",
        threshold=0.1,
        skip_cry_check=False,
        user_context=user_context or "",
        openai_model="gpt-4o-mini",
        disable_llm=False,
    )

    if not result.get("is_cry", False):
        msg = "No baby cry detected. Maybe try holding the mic closer to the baby."
        return (
            msg,
            gr.update(value=[], visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            json.dumps(result, indent=2),
            gr.update(visible=True),
        )

    probs = result.get("final_probabilities") or result.get("probabilities") or {}
    table = _format_probabilities_table(probs)
    user_summary = result.get("llm_user_summary", "")

    message = "Cry detected. Here are the estimated probabilities by category."
    disclaimer_text = (
        "Disclaimer: These probabilities are estimates and may be wrong. "
        "They do not cover more serious possibilities. If in doubt, speak to a medical professional."
    )
    return (
        message,
        gr.update(value=table, visible=True),
        gr.update(value=disclaimer_text, visible=True),
        gr.update(value=user_summary, visible=True),
        json.dumps(result, indent=2),
        gr.update(visible=True),
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <h1 style="font-size: 2.75rem; line-height: 1.2; margin: 0.25em 0 0.1em;">Why Baby Cry?</h1>
    """)
    gr.Markdown("Upload or record an audio clip of your baby, add some optional context, and click Analyse Cry.")

    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Upload or Record Audio (WAV recommended)")
        ctx = gr.Textbox(lines=4, label="Additional Context (optional)", placeholder="e.g., baby just ate, skipped nap")

    analyze_btn = gr.Button("Analyse Cry", variant="primary")
    try_again_btn = gr.Button("Try another cry", variant="secondary", visible=False)

    status = gr.Markdown(label="Status")
    probs_df = gr.Dataframe(headers=["Cry Type", "Probability"], label="Probabilities", row_count=(0, "dynamic"), col_count=2, visible=False)
    user_summary = gr.Markdown(label="Summary", visible=False)
    disclaimer = gr.Markdown(label="", visible=False)
    raw_json = gr.Code(label="Raw Output (debug)", language="json")

    analyze_btn.click(
        fn=analyze,
        inputs=[audio, ctx],
        outputs=[status, probs_df, disclaimer, user_summary, raw_json, try_again_btn],
    )

    # Reset UI to initial state
    def _reset():
        return (
            None,  # audio clear
            "",    # context clear
            "",    # status clear
            gr.update(value=[], visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            "",    # raw json clear
            gr.update(visible=False),  # hide try again
        )

    try_again_btn.click(
        fn=_reset,
        inputs=[],
        outputs=[audio, ctx, status, probs_df, user_summary, disclaimer, raw_json, try_again_btn],
    )


if __name__ == "__main__":
    demo.launch()


