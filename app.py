import runpod
from ctransformers import AutoModelForCausalLM

# Load quantized LLaMA model
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF",
    model_path="/app/models/llama-2-7b-chat.Q4_K_S.gguf",
    gpu_layers=20
)

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
            output = llm(prompt)
            responses.append(output.strip())

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
