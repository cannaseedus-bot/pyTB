# π--TB

## Binary-first ingest for MATRIX / ATOMIC-DOM

Below is the canonical binary-ingest model and a minimal packer for π-LM-style pipelines on i7-4790S-class hardware. The key idea is to do **text → numbers once, offline**, then only stream binary atoms in the hot loop.

### Architectural decision (locked)

Binary-first ingest is the correct choice for π-LM on i7-4790S-class hardware:

* JSON / HTML parsing is branch-heavy and cache-hostile.
* SIMD math is predictable and cache-friendly.
* The CPU is compute-capable but front-end bound.

So the winning move is: **do text → numbers once, offline, then never parse again.**

### Canonical pipeline

```
[ HTML | JSON | MD ]
        ↓ (one-time)
   CLEAN + NORMALIZE
        ↓
     TOKENIZE (π / tokenizer / symbol map)
        ↓
   PACK → BINARY ATOMS
        ↓
  mmap / seek / stream
        ↓
   π-LM / Embedding / Geometry
```

### ATOMIC-DOM binary rules (simple & fast)

**Atomic constraints**

* Fixed-width tokens (uint16 or uint32)
* Aligned blocks
* Sequential layout
* Stateless reads

Example:

* 65k vocab → `uint16`
* 32-byte alignment → AVX2-friendly
* Atom = `N` tokens (e.g. 256 / 512)

This yields predictable cache lines, fast `seek()`, and zero decode overhead.

### Minimal binary packer (drop-in)

`binary_pack.py`:

```python
import json
from pathlib import Path

import numpy as np

# ---- CONFIG ----
VOCAB_SIZE = 65536          # uint16
DTYPE = np.uint16
ATOM_SIZE = 256             # tokens per atom
OUT_FILE = "matrix_atoms.bin"

# ---- PLACEHOLDERS (plug your real ones in) ----
def load_and_clean(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix == ".json":
        try:
            obj = json.loads(text)
            text = json.dumps(obj, separators=(",", ":"))
        except ValueError:
            pass

    # minimal HTML stripping (replace later if needed)
    text = text.replace("<", " ").replace(">", " ")
    return text

def pi_tokenize(text: str):
    # TEMP: replace with π tokenizer / symbol mapper
    # deterministic integer mapping
    return [ord(c) % VOCAB_SIZE for c in text]

# ---- PACKER ----
def pack_directory(input_dir: str, out_file: str) -> None:
    tokens = []

    for path in Path(input_dir).rglob("*"):
        if path.suffix.lower() in (".txt", ".md", ".html", ".json"):
            text = load_and_clean(path)
            toks = pi_tokenize(text)
            tokens.extend(toks)

    # Pad to atom boundary
    pad = (-len(tokens)) % ATOM_SIZE
    if pad:
        tokens.extend([0] * pad)

    arr = np.array(tokens, dtype=DTYPE)
    arr.tofile(out_file)

    print(f"[OK] Packed {len(arr)} tokens")
    print(f"[OK] Atoms: {len(arr) // ATOM_SIZE}")
    print(f"[OK] Output: {out_file}")

if __name__ == "__main__":
    pack_directory("datasets", OUT_FILE)
```

### Runtime side (zero parsing, zero copy)

Memory-map the file:

```python
import numpy as np

ATOM_SIZE = 256
data = np.memmap(
    "matrix_atoms.bin",
    dtype=np.uint16,
    mode="r",
)

def read_atom(i: int):
    start = i * ATOM_SIZE
    return data[start:start + ATOM_SIZE]
```

### Why this fits π-LM perfectly

* π-LM wants deterministic streams.
* ATOMIC-DOM wants fixed geometry.
* MATRIX wants binary invariants.

This yields reproducible inference, stable geometry, future WebGPU compatibility, and GGUF / embedding interop on the same binary substrate.
