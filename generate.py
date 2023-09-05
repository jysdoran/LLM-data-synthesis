import re
from pathlib import Path
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

from tqdm import tqdm

from data import CodeSearchAdvExample, CodeSearchAdvDataset, JSONLDataset
from tokenise_fp import function_data_from_string

openai_key_path = Path("/home/james/.config/.keys/openai")
with openai_key_path.open("r") as f:
    OPENAI_KEY = f.read().strip()

TEMPLATE_DOC_TO_CODE = """
Write a python function that has this as a docstring:
\"\"\"
{docstring}
\"\"\"
Do not reply with any text other than this function.
"""

SYSTEM_MESSAGE_D2C = SystemMessage(
    content="You are a helpful python programming assistant. "
    "I will provide a docstring and you should write a python function that has that docstring. "
    "Do not reply with any text other than this function."
)

SYSTEM_MESSAGE_C2D = SystemMessage(
    content="You are a helpful python programming assistant. "
    "I will provide a python function and you should write a succinct docstring for that python function. "
    "Do not reply with any text other than this docstring."
)

SYSTEM_MESSAGE_HARDNEGATIVE = SystemMessage(
    content="You are a helpful python programming assistant. "
    "I will provide a python function and you should write a superficially similar but semantically very different python function and docstring."
    "Do not reply with any text other than this function."
)

SYSTEM_MESSAGE_HARDPOSITIVE = SystemMessage(
    content="You are a helpful python programming assistant. "
    "I will provide a python function and you should heavily refactor it (renaming variables, reordering statements, etc.) "
    "into a very different looking but semantically identical python function with a new docstring."
    "Do not reply with any text other than this function."
)

DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+"
)


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [
        t
        for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr)
        if t is not None and len(t) > 0
    ]


class SyntheticDataGenerator:
    def __init__(self, chat_model: ChatOpenAI):
        self.chat_model = chat_model

    def generate_code_from_docstrings(self, docstrings: List[str]):
        docstring_message = HumanMessagePromptTemplate.from_template(
            '"""{docstring}"""',
        )
        prompt = ChatPromptTemplate.from_messages(
            [SYSTEM_MESSAGE_D2C, docstring_message]
        )
        generated_data = []
        mismatches = []
        for docstring in tqdm(docstrings):
            message = None
            try:
                # Add vague docstring length to max tokens to compensate for including the docstring
                message = self.chat_model.predict_messages(
                    messages=prompt.format_messages(docstring=docstring),
                    max_tokens=len(docstring.split()) + 300,
                )
                function_data = function_data_from_string(message.content)

                if not (function_data.get("function_tokens") and function_data.get("docstring_tokens")):
                    raise ValueError("Generated function data has empty function or docstring")
            except Exception as e:
                tqdm.write(str(e))
                function_data = {}
                if message is not None:
                    function_data["message"] = message.content
                    function_data["error"] = str(e)

            if function_data.get("docstring") != docstring:
                mismatches.append((function_data, docstring))

            generated_data.append(function_data)

        return generated_data, mismatches

    def generate_docstrings_from_code(self, codes: List[str]):
        code_message = HumanMessagePromptTemplate.from_template(
            "{code}",
        )
        prompt = ChatPromptTemplate.from_messages([SYSTEM_MESSAGE_C2D, code_message])
        generated_data = []
        for code in tqdm(codes):
            message = None
            try:
                message = self.chat_model.predict_messages(
                    messages=prompt.format_messages(code=code),
                    max_tokens=400,
                )
                docstring_data = {
                    "docstring": message.content,
                    "docstring_tokens": tokenize_docstring_from_string(message.content),
                }

            except Exception as e:
                tqdm.write(str(e))
                docstring_data = {}
                if message is not None:
                    docstring_data["message"] = message.content
                    docstring_data["error"] = str(e)

            generated_data.append(docstring_data)

        return generated_data

    def generate_hard_examples(self, codes: List[str], negative=True):
        code_message = HumanMessagePromptTemplate.from_template(
            "{code}",
        )
        system_message_hard = (
            SYSTEM_MESSAGE_HARDNEGATIVE if negative else SYSTEM_MESSAGE_HARDPOSITIVE
        )
        prompt = ChatPromptTemplate.from_messages([system_message_hard, code_message])
        generated_data = []
        for code in tqdm(codes):
            message = None
            try:
                # Add vague docstring length to max tokens to compensate for including the docstring
                message = self.chat_model.predict_messages(
                    messages=prompt.format_messages(code=code),
                    max_tokens=2000,
                )
                # tqdm.write("=====")
                # tqdm.write(message.content)
                # tqdm.write(code)

                function_data = function_data_from_string(message.content)

                if not (function_data.get("function_tokens") and function_data.get("docstring_tokens")):
                    raise ValueError("Generated function data has empty function or docstring")
            except Exception as e:
                tqdm.write(str(e))
                function_data = {}
                if message is not None:
                    function_data["message"] = message.content
                    function_data["error"] = str(e)

            generated_data.append(function_data)

        return generated_data


def generate_data_chat_gpt():
    mode = "hardpositive"
    original_dataset = CodeSearchAdvDataset()
    start = 6400
    end = 12800
    original_dataset.load_jsonl(start=start, end=end)

    chat_gpt = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-3.5-turbo")
    data_generator = SyntheticDataGenerator(chat_gpt)

    if mode == "d2c":
        data_dicts, mismatches = data_generator.generate_code_from_docstrings(
            [e.docstring for e in original_dataset]
        )
    elif mode == "c2d":
        dedocstringed_code = []
        data_dicts = []
        for e in original_dataset:
            code = e.code
            code = code.replace(e.docstring, "")
            if code == e.code:
                raise ValueError(f"No docstring found in code: {code}")
            dedocstringed_code.append(code)
            data_dicts.append(e.__dict__)

        docstring_dicts = data_generator.generate_docstrings_from_code(
            dedocstringed_code
        )

        for data_dict, docstring_dict in zip(data_dicts, docstring_dicts):
            data_dict.update(docstring_dict)
    elif "hard" in mode:
        data_dicts = data_generator.generate_hard_examples(
            [e.code for e in original_dataset], negative="negative" in mode
        )

    synth_data_path = Path(f"./datasets/{mode}_semisynthetic.jsonl")

    semisynthetic_jsonl = JSONLDataset(synth_data_path)
    semisynthetic_jsonl.update_range(data_dicts, start=start, end=end)

    semisynthetic_jsonl.save_jsonl(synth_data_path)

    if mode == "d2c":
        mismatches_path = Path(f"./datasets/{mode}_mismatches1.jsonl")
        with mismatches_path.open("w") as f:
            for mismatch in mismatches:
                f.write(str(mismatch) + "\n")


def fix_missing_data():
    original_dataset = CodeSearchAdvDataset()
    mode = "hardnegative"
    start = 0
    end = 12800
    original_dataset.load_jsonl(start=start, end=end)

    # model = "gpt-3.5-turbo-16k"
    model = "gpt-3.5-turbo"
    chat_gpt = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name=model)
    data_generator = SyntheticDataGenerator(chat_gpt)

    synth_data_path = Path(f"./datasets/{mode}_semisynthetic.jsonl")

    semisynthetic_jsonl = JSONLDataset(synth_data_path)

    # These are examples that failed to generate code
    failed_idxs = list(
        filter(
            lambda i: not (bool(semisynthetic_jsonl[i].get("function_tokens"))
            and bool(semisynthetic_jsonl[i].get("docstring_tokens"))),
            range(start, end),
        )
    )
    # failed_idxs = list(
    #     filter(
    #         lambda i: "function_tokens" not in semisynthetic_jsonl[i].keys(),
    #         range(start, end),
    #     )
    # )
    # Identify identical functions
    # failed_idxs = list(
    #     filter(
    #         lambda i: "function_tokens" not in semisynthetic_jsonl[i] or semisynthetic_jsonl[i]["function_tokens"] == original_dataset[i].code_tokens,
    #         range(start, end),
    #     )
    # )
    print("Failed idxs:", failed_idxs)
    # Failed idxs: [271, 275, 369, 656, 970, 1979, 2549, 2919, 3538, 4492, 4620, 4621, 5037, 5142, 5152, 5227, 5400, 5416, 5431, 5814]
    # Failed idxs: [6518, 6519, 6521, 6752, 7117, 7687, 8362, 8397, 8579, 8585, 8728, 8945, 8947, 8973, 9034, 9090, 9264, 9381, 9572, 10630, 10823, 11024, 11563, 11577, 11578, 11993, 12277]

    if mode == "d2c":
        # docstrings = [" ".join(original_dataset[idx].docstring_tokens) for idx in failed_idxs]
        docstrings = [original_dataset[idx].docstring for idx in failed_idxs]

        data_dicts, mismatches = data_generator.generate_code_from_docstrings(docstrings)

        # for idx, new_example in zip(failed_idxs, data_dicts):
        #     if not new_example.get("docstring_tokens"):
        #         new_example["docstring"] = original_dataset[idx].docstring
        #         new_example["docstring_tokens"] = original_dataset[idx].docstring_tokens
        #         new_example["docstring_summary"] = " ".join(original_dataset[idx].docstring_tokens)

        mismatches_path = Path(f"./datasets/{mode}_retry_mismatches.jsonl")
        with mismatches_path.open("w") as f:
            for mismatch in mismatches:
                f.write(str(mismatch) + "\n")

    elif mode == "c2d":
        failed_codes = []
        for idx in failed_idxs:
            example = original_dataset[idx]
            code = example.code.replace(
                example.docstring, " ".join(example.docstring_tokens)
            )
            failed_codes.append(code)

        data_dicts = data_generator.generate_docstrings_from_code(failed_codes)
    elif "hard" in mode:
        negative = "negative" in mode
        parse_from_message = False
        failed_codes = []
        data_dicts = []
        for idx in failed_idxs:
            example = original_dataset[idx]
            code = example.code
            code = code.replace(example.docstring, " ".join(example.docstring_tokens))
            failed_codes.append(code)

            if parse_from_message:
                try:
                    code_match = re.search(
                        r"(?:```)?(?:python)?\s*([^`]+)(?:```)?\s*$",
                        semisynthetic_jsonl[idx]["message"],
                    )
                    data_dicts.append(
                        function_data_from_string(code_match.group(1).strip())
                    )
                except Exception as e:
                    tqdm.write(str(e))
                    data_dicts.append(semisynthetic_jsonl[idx])

        if not parse_from_message:
            data_dicts = data_generator.generate_hard_examples(
                failed_codes, negative=negative
            )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    for idx, new_example in zip(failed_idxs, data_dicts):
        semisynthetic_jsonl[idx] = new_example

    semisynthetic_jsonl.save_jsonl(synth_data_path)


def main():
    # generate_data_chat_gpt()
    fix_missing_data()


if __name__ == "__main__":
    main()
