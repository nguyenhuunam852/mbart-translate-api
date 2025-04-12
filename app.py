from fastapi import FastAPI, Request
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import uvicorn

app = FastAPI()
model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

@app.post("/translate")
async def translate(request: Request):
    data = await request.json()
    texts = data["texts"]
    src_lang = data["src_lang"]
    tgt_lang = data["tgt_lang"]

    tokenizer.src_lang = src_lang
    encoded = tokenizer(texts, return_tensors="pt")
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    return {"translation": translated}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
