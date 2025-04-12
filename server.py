from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

@app.post("/test")
async def test(request: Request):
    try:
        input_data = await request.json()
        texts = input_data.get("input").get("texts")
        src_lang = input_data.get("input").get("src_lang")
        tgt_lang = input_data.get("input").get("tgt_lang")

        if not all([texts, src_lang, tgt_lang]):
            return {
                "status": "error",
                "message": "Missing required fields: texts, src_lang, tgt_lang"
            }

        tokenizer.src_lang = src_lang
        print(texts)
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        
        
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {
            "status": "success",
            "output": {"translation": translated}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)