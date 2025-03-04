import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from llama_cpp import Llama

class InferlessPythonModel:
    def initialize(self):
        model_id = "unsloth/phi-4-GGUF"
        snapshot_download(repo_id=model_id,allow_patterns=["phi-4-Q4_K_M.gguf"])
        self.llm = Llama.from_pretrained(repo_id=model_id,filename="phi-4-Q4_K_M.gguf",main_gpu=0,n_gpu_layers=-1)
      
    def infer(self, inputs,stream_output_handler):
        prompt = inputs["prompt"]
        system_prompt = inputs.get("system_prompt","You are a friendly bot.")
        temperature = inputs.get("temperature",0.7)
        top_p = inputs.get("top_p",0.1)
        top_k = inputs.get("top_k",40)
        repeat_penalty = inputs.get("repeat_penalty",1.18)
        max_tokens = inputs.get("max_tokens",256)
        
        output = self.llm.create_chat_completion(
                    messages = [
                      {"role": "system", "content": f"{system_prompt}"},
                      {"role": "user","content": f"{prompt}"}],
                    temperature=temperature, top_p=top_p, top_k=top_k,repeat_penalty=repeat_penalty,max_tokens=max_tokens,
                    stream=True
        )
        for token_data in output:
            delta = token_data["choices"][0].get("delta", {})
            token_text = delta.get("content", "")
            full_response += token_text
            if token_text:
                output_dict = {}
                output_dict["OUT"] = token_text
                stream_output_handler.send_streamed_output(output_dict)
        
        stream_output_handler.finalise_streamed_output()
        
    def finalize(self):
        self.llm = None
