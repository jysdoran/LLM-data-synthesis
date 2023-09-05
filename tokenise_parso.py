from typing import NamedTuple, List
import re

import parso

IS_WHITESPACE_REGEX = re.compile(r"\s+")

DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+"
)


def tokenize_docstring_from_string(docstr: str) -> List[str]:
    return [
        t
        for t in DOCSTRING_REGEX_TOKENIZER.findall(docstr)
        if t is not None and len(t) > 0
    ]


class ParsedCode(NamedTuple):
    code_tokens: List[str]
    comment_tokens: List[str]


def tokenize_python_from_string(
    code: str,
    func_only: bool = True,
    report_errors: bool = False,
    only_ids: bool = False,
    add_keywords: bool = True,
    keep_comments = True,
) -> ParsedCode:
    """
    Tokenize Python code given a string.

    Args:
        code: The input code
        func_only: if you want to only parse functions in code.
        report_errors: Flag that turns on verbose error reporting
        only_ids: Return only the identifiers within the code
        add_keywords: Return keywords (used only when only_ids=True)

    Returns:
        Pair of lists. First list is sequence of code tokens; second list is sequence of tokens in comments.
    """
    try:
        try:
            parsed_ast = parso.parse(code, error_recovery=False, version="2.7")
        except parso.parser.ParserSyntaxError:
            parsed_ast = parso.parse(code, error_recovery=False, version="3.7")

        code_tokens, comment_tokens = [], []

        func_nodes = list(parsed_ast.iter_funcdefs())

        # parse arbitrary snippets of code that are not functions if func_only = False
        if not func_only:
            func_nodes = [parsed_ast]

        for (
            func_node
        ) in func_nodes:  # There should only be one, but we can process more...
            doc_node = func_node.get_doc_node()
            leaf_node = func_node.get_first_leaf()
            while True:
                # Skip over the docstring:
                if leaf_node is doc_node:
                    leaf_node = leaf_node.get_next_leaf()

                # First, retrieve comment tokens:
                for prefix in leaf_node._split_prefix():
                    if prefix.type == "comment":
                        comment_text = prefix.value[1:]  # Split off the leading "#"
                        comment_tokens.extend(
                            tokenize_docstring_from_string(comment_text)
                        )
                        if keep_comments:
                            code_tokens.append(prefix.value)

                # Second, stop if we've reached the end:
                if leaf_node.type == "endmarker":
                    break

                # Third, record code tokens:
                if not (IS_WHITESPACE_REGEX.match(leaf_node.value)):
                    if only_ids:
                        if leaf_node.type == "name":
                            code_tokens.append(leaf_node.value)
                    else:
                        if leaf_node.type == "keyword":
                            if add_keywords:
                                code_tokens.append(leaf_node.value)
                        elif leaf_node.type == "operator":
                            if leaf_node.value == "**":
                                code_tokens.extend(["*", "*"])
                            else:
                                code_tokens.append(leaf_node.value)
                        else:
                            code_tokens.append(leaf_node.value)
                leaf_node = leaf_node.get_next_leaf()
        return ParsedCode(code_tokens=code_tokens, comment_tokens=comment_tokens)
    except Exception as e:
        if report_errors:
            print("Error tokenizing: %s" % (e,))
        return ParsedCode(code_tokens=[], comment_tokens=[])


def validate_tokenization():
    from data import CodeSearchAdvDataset

    dataset = CodeSearchAdvDataset()
    dataset.load_jsonl()

    for i in range(len(dataset)):
        example = dataset[i]
        if i == 32:
            print(32)
        parsed_code = tokenize_python_from_string(example.code, report_errors=True)

        if parsed_code.code_tokens != example.code_tokens:
            print("=======Mismatch=======", example.idx)
            print(parsed_code.code_tokens)
            print(example.code_tokens)

# ['def', 'change_return_type', '(', 'f', ')', ':', '@', 'wraps', '(', 'f', ')', 'def', 'wrapper', '(', '*', 'args', ',', '**', 'kwargs', ')', ':', 'if', 'kwargs', '.', 'has_key', '(', "'return_type'", ')', ':', 'return_type', '=', 'kwargs', '[', "'return_type'", ']', 'kwargs', '.', 'pop', '(', "'return_type'", ')', 'return', 'return_type', '(', 'f', '(', '*', 'args', ',', '**', 'kwargs', ')', ')', 'elif', 'len', '(', 'args', ')', '>', '0', ':', 'return_type', '=', 'type', '(', 'args', '[', '0', ']', ')', 'return', 'return_type', '(', 'f', '(', '*', 'args', ',', '**', 'kwargs', ')', ')', 'else', ':', 'return', 'f', '(', '*', 'args', ',', '**', 'kwargs', ')', 'return', 'wrapper'],
# ['def', 'change_return_type', '(', 'f', ')', ':', '@', 'wraps', '(', 'f', ')', 'def', 'wrapper', '(', '*', 'args', ',', '*', '*', 'kwargs', ')', ':', 'if', 'kwargs', '.', 'has_key', '(', "'return_type'", ')', ':', 'return_type', '=', 'kwargs', '[', "'return_type'", ']', 'kwargs', '.', 'pop', '(', "'return_type'", ')', 'return', 'return_type', '(', 'f', '(', '*', 'args', ',', '*', '*', 'kwargs', ')', ')', 'elif', 'len', '(', 'args', ')', '>', '0', ':', 'return_type', '=', 'type', '(', 'args', '[', '0', ']', ')', 'return', 'return_type', '(', 'f', '(', '*', 'args', ',', '*', '*', 'kwargs', ')', ')', 'else', ':', 'return', 'f', '(', '*', 'args', ',', '*', '*', 'kwargs', ')', 'return', 'wrapper'])

# ['def', '_nama', '(', 'self', ')', ':', 'hasil', '=', 'self', '.', 'nama', 'if', 'self', '.', 'nomor', ':', 'hasil', '+=', '" [{}]"', '.', 'format', '(', 'self', '.', 'nomor', ')', 'if', 'self', '.', 'kata_dasar', ':', 'hasil', '=', '" » "', '.', 'join', '(', 'self', '.', 'kata_dasar', ')', '+', '" » "', '+', 'hasil', 'return', 'hasil']
# ['def', '_nama', '(', 'self', ')', ':', 'hasil', '=', 'self', '.', 'nama', 'if', 'self', '.', 'nomor', ':', 'hasil', '+=', '" [{}]"', '.', 'format', '(', 'self', '.', 'nomor', ')', 'if', 'self', '.', 'kata_dasar', ':', 'hasil', '=', '" » ".', 'j', 'oin(', 's', 'elf.', 'k', 'ata_dasar)', ' ', ' ', ' » " +', 'h', 'sil', 'return', 'hasil'])


if __name__ == "__main__":
    validate_tokenization()
