import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# System prompt
system_prompt = "trả lời tối đa 20 chữ.\n"
system_prompt += (
    "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. "
    "Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    'Viet-Mistral/Vistral-7B-Chat',
    token=os.getenv("secret")
)

model = AutoModelForCausalLM.from_pretrained(
    'Viet-Mistral/Vistral-7B-Chat',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=True,
    token=os.getenv("secret")
)

# Handler function
def handler(event):
    try:
        input_data = event.get("input", {})
        texts = input_data.get("texts")

        if not texts:
            return {
                "status": "error",
                "message": "Missing required field: texts"
            }

        responses = []
        for prompt in texts:
            conversation = [
                {"role": "user", "content": system_prompt + "\n" + "trả lời câu hỏi:" + prompt}
            ]

            input_ids = tokenizer.apply_chat_template(
                conversation, return_tensors="pt"
            ).to(model.device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id
            )

            assistant = tokenizer.decode(
                output_ids[0, input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            responses.append(assistant)
            
            conversation.append({"role": "assistant", "content": assistant})

        return {
            "status": "success",
            "output": {"responses": responses}
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Start serverless handler
runpod.serverless.start({"handler": handler})
