"""Lab 07 — peek inside a .gguf: it's just a header + metadata + tensors.
Run: python labs/lab07_gguf_teardown.py export/tinystories-slm-Q8_0.gguf
"""
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else "export/tinystories-slm-Q8_0.gguf"
with open(path, "rb") as f:
    magic = f.read(4)
    version, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))
print(f"magic        : {magic}  (should be b'GGUF')")
print(f"version      : {version}")
print(f"tensor count : {n_tensors}")
print(f"metadata kv  : {n_kv}")
print("\nThe header advertises how many tensors (the weights) and how many")
print("metadata entries (arch, hyperparams, tokenizer) follow. A .gguf is a")
print("self-describing container: the runtime reads this to know how to run it.")
