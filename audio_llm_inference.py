import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from prompter import llama_v2_prompt, extract_text_llama2


class AudioTextTranscriptionAnalysis:
    _instance = None
    selected_gpu = 2

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        # Usage
        model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
        model_basename = "meta-llama/Llama-2-7b-chat-hf"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            device_map=self.selected_gpu
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_basename,
            quantization_config=quantization_config,
            device_map=self.selected_gpu
        )
        self.tokenizer.add_special_tokens({
            "eos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.eos_token_id),
            "bos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.bos_token_id),
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def generate_text(self, prompt):
        try:
            with torch.inference_mode():
                encoding = self.tokenizer(prompt,
                                          truncation=True,
                                          padding=True,
                                          return_tensors="pt").to(f"cuda:{self.selected_gpu}")
                output = self.model.generate(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    max_new_tokens=700,
                    do_sample=True,
                    temperature=0.2,
                    top_k=20,
                    top_p=0.96,
                )
                res = self.tokenizer.decode(
                    output[0], skip_special_tokens=False)
                extracted_text = extract_text_llama2(res)
                return extracted_text
        except Exception as e:
            print(e)
            return f"Error {e}"

    def transcript_qa(self, chroma_processor, prompt):
        # Assuming you have a proper instance of ChromaProcessor
        # docs = chroma_processor.db.similarity_search(prompt)
        print("Prompt: " + prompt)
        docs = chroma_processor.search(prompt, k=5)
        res = ' '.join([str(elem.page_content) for elem in docs])
        completion = llama_v2_prompt(
            [{"role": "user", "content": prompt, "context": res}])
        print("Full prompt: " + prompt + "\n")
        print("Completion: " + completion + "\n")
        # Extracting text after "Completion:"
        return self.generate_text(completion)
