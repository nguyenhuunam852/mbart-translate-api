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
    device_map={"": 0},
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            input_ids = tokenizer.apply_chat_template(
                conversation, return_tensors="pt"
            ).to(model.device)

            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=768,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                temperature=0.1,
                repetition_penalty=1.05,
            )

            assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip() 
            print("Assistant: ", assistant) 
            
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
