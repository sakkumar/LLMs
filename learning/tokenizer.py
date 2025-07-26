from importlib.metadata import version
import torch
import tiktoken

print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text=("hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
unknown="Akwirw ier"
unknownintegers=tokenizer.encode(unknown)
print(unknownintegers)
unknownstrings = tokenizer.decode(unknownintegers)
print(unknownstrings)
