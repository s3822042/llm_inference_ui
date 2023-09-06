# implement Prompter class
import re

def llama_v2_prompt(messages: list[dict]):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = """
    You are a helpful and friendly assistant. You are given an transcription of an audio recording. 
    Act as a professional transcriptor and answer the question based on the transcription. The question is follow the following sample
    User: Hi, What is the conversation is about ?
    Assistant: Hi, thankyou for asking. Based on the content of the recording , the anwser to the question (question) is (answer)
    Here is the transcription of the audio recording:
    {context}
    """

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT.format(context=messages[0]["context"]),
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def llama_1_prompt(messages: dict):
    system = "Bạn là một trợ lý an toàn và thân thiện.Hãy đưa ra câu trả lời phù hợp dựa vào hướng dẫn được cung cấp dựa vào ngữ cảnh sau đây:"
    context_prefix = "\n ### Context: "
    question_prefix = "\n### Question: "
    answer_prefix = "\n### Answer:"

    return system + context_prefix + messages["context"] + question_prefix + messages["content"] + answer_prefix

def extract_text_llama2(input_string):
        start_pattern = r"\[/INST\]\s*(.*?)<\/s>"
        match = re.search(start_pattern, input_string, re.DOTALL)

        if match:
            extracted_text = match.group(1).replace("<|END|>", "")
            return extracted_text
        else:
            return None

# extract text llama 1 get the first sentence between ###Answer till the next ###Answer
def extract_text_llama1(input_string):
    pattern = r'### Answer:\s+(.*?)\s*(?:###|$)'
    matches = re.findall(pattern, input_string, re.DOTALL)

    if matches:
        extracted_text = matches[0].replace("<|END|>", "")
        return extracted_text
    else:
        return None
