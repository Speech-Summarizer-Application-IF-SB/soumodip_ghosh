# summarizer.py
"""
Summarizer Module
-----------------
This script takes a diarized transcript or plain meeting text as input
and generates a concise summary using a pre-trained Hugging Face model (T5-small).

Usage:
    python summarizer.py "Your meeting transcript text here"
"""

import sys
from transformers import pipeline

def summarize_text(text: str, model_name: str = "t5-small", max_length: int = 120, min_length: int = 25) -> str:
    """
    Summarizes a given text using a transformer model.

    Args:
        text (str): Input text (transcript or notes).
        model_name (str): Hugging Face model to use.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: Generated summary.
    """
    try:
        summarizer = pipeline("summarization", model=model_name)
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"[Error during summarization] {e}"

if __name__ == "__main__":
    # If user passes text directly from terminal
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        # If no argument provided, use a sample text
        input_text = (
            "The team discussed quarterly performance and agreed that revenue "
            "should increase by 20% next quarter through new marketing initiatives. "
            "They also reviewed customer feedback and product updates."
        )

    print("\n--- Original Text ---")
    print(input_text)
    print("\n--- Generated Summary ---")
    print(summarize_text(input_text))
