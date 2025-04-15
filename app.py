import runpod
# Load quantized LLaMA model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
system_prompt = "trả lời tối đa 20 chữ.\n"
system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat',token = os.getenv("secret"))
model = AutoModelForCausalLM.from_pretrained(
    'Viet-Mistral/Vistral-7B-Chat',
    torch_dtype=torch.bfloat16,
    device="cuda",
    token = os.getenv("secret")
)

conversation = [{"role": "system", "content": system_prompt}]

def handler(event):
    """
    RunPod serverless handler function.
    Expects input event with 'texts' (a list of prompts).
    Returns model-generated completions.
    """
    try:
        # Extract input
        input_data = event.get("input", {})
        texts = input_data.get("texts")

        if not texts:
            return {
                "status": "error",
                "message": "Missing required field: texts"
            }

        # Process each prompt
        responses = []
        for prompt in texts:
            conversation.append({"role": "user", "content": prompt})
            input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to('cuda')
            
            generated_ids = input_ids
            
            for _ in range(768): 
                next_token_logits = model(generated_ids).logits[:, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                if token_text:
                    print(token_text+' ', end='', flush=True)
    
            assistant = tokenizer.decode(generated_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
            responses.append(assistant)

        return {
            "status": "success",
            "output": {"responses": responses}
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
