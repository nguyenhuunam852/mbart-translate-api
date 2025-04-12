import runpod
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Initialize model and tokenizer globally to avoid reloading per request
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def handler(event):
    """
    RunPod serverless handler function.
    Expects input event with 'texts', 'src_lang', and 'tgt_lang'.
    Returns translated text.
    """
    try:
        # Extract input from event
        input_data = event.get("input", {})
        texts = input_data.get("texts")
        src_lang = input_data.get("src_lang")
        tgt_lang = input_data.get("tgt_lang")

        # Validate inputs
        if not all([texts, src_lang, tgt_lang]):
            return {
                "status": "error",
                "message": "Missing required fields: texts, src_lang, tgt_lang"
            }

        # Set source language and encode input
        tokenizer.src_lang = src_lang
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        # Generate translation
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id.get(tgt_lang)
        )

        # Decode output
        translated = tokenizer.decode(generated, skip_special_tokens=True)

        # Return RunPod-compatible response
        return {
            "status": "success",
            "output": {"translation": translated}
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Register the handler with RunPod
runpod.serverless.start({"handler": handler})