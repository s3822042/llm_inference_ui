import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


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

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            f"cuda:{self.selected_gpu}")
        with torch.inference_mode():
            tokens = self.model.generate(
                **inputs,
                do_sample=True,
                top_k=10,  # (int, optional, defaults to 50)
                top_p=0.95,  # (float, optional, defaults to 1.0)
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=1000,
            )
            completion_tokens = tokens[0][inputs['input_ids'].size(1):]
            completion = self.tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            return completion

    def transcript_qa(self, chroma_processor, prompt):
        # Assuming you have a proper instance of ChromaProcessor
        # docs = chroma_processor.db.similarity_search(prompt)
        print("Prompt: " + prompt)
        docs = chroma_processor.search(prompt, k=5)
        res = ' '.join([str(elem.page_content) for elem in docs])
        # completion = self.generate_text(f"""
        # You are a safe and helpful assistant. Now act as an audio-text transcription analyst.
        # Generate the output that an audio-text transcription analyst may give.
        # You are given the following audio transcription. Only answer the question that could only be found within the audio transcription
        # {res}
        # {prompt}
        # """)
        full_prompt = f"""
        You are a helpful assistant in the medical field who gives briefing about conversational recordings. 
        You will be given a recording for context and a question. 
        You answer the question solely based on the recordings and avoid disclaimers.
        The question: "{prompt}"
        Conversation recording: "{res}"
        Your answer: 
        """
        completion = self.generate_text(full_prompt)
        print("Full prompt: " + full_prompt + "\n")
        print("Completion: " + completion + "\n")

        # Extracting text after "Completion:"
        return completion
