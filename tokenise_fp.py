import function_parser
import os

from function_parser.language_data import LANGUAGE_METADATA
from function_parser.process import DataProcessor
from tree_sitter import Language
from tqdm import tqdm

language = "python"
DataProcessor.PARSER.set_language(
    Language(
        os.path.join(function_parser.__path__[0], "tree-sitter-languages.so"), language
    )
)
processor = DataProcessor(
    language=language, language_parser=LANGUAGE_METADATA[language]["language_parser"]
)

def function_data_from_string(code):
    tree = DataProcessor.PARSER.parse(code.encode())
    try:
        functions = processor.language_parser.get_definition(tree, code)
        function = max(functions, key=lambda f: len(f["function_tokens"]))
        if not function["function_tokens"]:
            raise ValueError(f"Could not parse function from code: {code}")
    except IndexError as e:
        raise ValueError(f"Could not parse function from code: {code}") from e

    data_dict = processor.extract_function_data(function, "", "", "")
    del data_dict["url"]
    del data_dict["nwo"]
    del data_dict["sha"]
    del data_dict["path"]
    return data_dict


def validate_tokenization():
    from data import CodeSearchAdvDataset

    dataset = CodeSearchAdvDataset()
    dataset.load_jsonl()
    print("Dataset Loaded - Validating...")

    mismatches = []
    for i in tqdm(range(len(dataset))):
        example = dataset[i]

        function_data = function_data_from_string(example.code)

        if function_data["function_tokens"] != example.code_tokens:
            tqdm.write(f"=======Mismatch=======[{example.idx}]")
            tqdm.write(str(function_data["function_tokens"]))
            tqdm.write(str(example.code_tokens))
            mismatches.append(
                (example.idx, function_data["function_tokens"], example["code_tokens"])
            )

    # Save mismatches to file
    with open("mismatches.txt", "w") as f:
        for mismatch in mismatches:
            f.write(str(mismatch) + "\n")


if __name__ == "__main__":
    validate_tokenization()
