import io 
import os
import json

# Sub-method

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

# Dump a str or dictionary to a file in json format
def jdump(obj, f, mode="w", indent=4, default=str, ensure_ascii=False):
    """ Dump a str or dictionary to a file in json format. 
    
    Args:
        obj: An object to be written.
        f:   A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str'.    
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {rtype(obj)}")
    f.close()

# Load a json file into a dictionary
def jload(f, mode="r"):
    """ Load a .json file into a dictionary """
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

# 日本語判定
def is_japanese(char):
    """
    指定された文字が日本語の文字であるかどうかを判定します。
    refers: https://zenn.dev/shundeveloper/articles/a4be0379508e2d

    Args:
        char (str): 判定する単一の文字。

    Returns:
        bool: 文字が日本語の場合はTrue、それ以外の場合はFalse。
    """
    return (
            '\u3040' <= char <= '\u309F' or  # Hiragana
        '\u30A0' <= char <= '\u30FF' or  # Katakana
        '\uFF65' <= char <= '\uFF9F' or  # Half-width Katakana
        '\u31F0' <= char <= '\u31FF' or  # Katakana Phonetic Extensions
        '\u4E00' <= char <= '\u9FFF' or  # CJK Unified Ideographs
        '\u3400' <= char <= '\u4DBF' or  # CJK Extension A
        '\u20000' <= char <= '\u2A6DF' or  # CJK Extension B
        '\u2A700' <= char <= '\u2B73F' or  # CJK Extension C
        '\u2B820' <= char <= '\u2CEAF' or  # CJK Extension E
        '\u2CEB0' <= char <= '\u2EBEF' or  # CJK Extension F
        '\u3000' <= char <= '\u303F'      # Japanese Punctuation    
    )
