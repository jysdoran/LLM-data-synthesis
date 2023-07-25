import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

DATASET_PATH = Path(
    "/home/james/Thesis/jysdoran-CodeXGLUE/Text-Code/NL-code-search-Adv/dataset"
)


@dataclass
class CodeSearchAdvExample:
    """A single training example for the CodeSearchAdv dataset."""

    idx: int
    func_name: str
    original_string: str
    code: str
    docstring: str
    code_tokens: List[str]
    docstring_tokens: List[str]

    def __init__(self, js):
        self.idx = js["idx"]
        self.func_name = js["func_name"]
        self.original_string = js["original_string"]
        self.code = js["code"]
        self.docstring = js["docstring"]
        self.code_tokens = js["code_tokens"]
        self.docstring_tokens = js["docstring_tokens"]


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.idx = idx


def convert_examples_to_features(js, tokenizer, args):
    # code
    if "code_tokens" in js:
        code = " ".join(js["code_tokens"])
    else:
        code = " ".join(js["function_tokens"])
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.block_size - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = " ".join(js["docstring_tokens"])
    nl_tokens = tokenizer.tokenize(nl)[: args.block_size - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.block_size - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js["url"], js["idx"])


def load_data_codesearch_adv(
    file_path=DATASET_PATH / "train.jsonl",
) -> List[CodeSearchAdvExample]:
    examples = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            assert js["language"] == "python"
            examples.append(CodeSearchAdvExample(js))

    return examples

    # if "train" in file_path:
    #     for idx, example in enumerate(self.examples[:3]):
    #         logger.info("*** Example ***")
    #         logger.info("idx: {}".format(idx))
    #         logger.info(
    #             "code_tokens: {}".format(
    #                 [x.replace("\u0120", "_") for x in example.code_tokens]
    #             )
    #         )
    #         logger.info("code_ids: {}".format(" ".join(map(str, example.code_ids))))
    #         logger.info(
    #             "nl_tokens: {}".format(
    #                 [x.replace("\u0120", "_") for x in example.nl_tokens]
    #             )
    #         )
    #         logger.info("nl_ids: {}".format(" ".join(map(str, example.nl_ids))))


def main():
    train = load_data_codesearch_adv(DATASET_PATH / "train.jsonl")
    # for i in range(3):
    #     print(train[i])
    print(len(train))


if __name__ == "__main__":
    main()
