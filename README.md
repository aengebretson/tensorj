# tj
TenorJ - a J programming language interface to tensorflow

---

## 🚀 **J is the Future of HPC and AI**

### 🧠 High-Level Abstraction with No Performance Compromise

J offers **array-oriented, functional programming** that expresses complex operations in **fewer lines of code** than any mainstream language. Its **tacit (point-free)** style avoids explicit loops and variables, enabling **automatic parallelism** and **optimization opportunities** at the interpreter level.

In contrast to Python's verbose procedural style—even with NumPy or TensorFlow—J expressions are **closer to how humans reason** about data transformation, not how machines must implement it.

---

### ⚡ Built for Parallelism, Not Retrofitted

Most modern languages (e.g., Python, C++) bolt on parallel computing. J was born from the mathematics of **arrays, ranks, and compositions**, which align directly with **tensor operations** and **multi-core/GPU execution models**. This makes J **a natural fit for automatic parallelization**, just like TensorFlow graphs—but more general and expressive.

In J, ideas like broadcasting, reductions, and convolutions **fall out of the language model**, not from special libraries or syntax extensions.

---

### ✍️ Notation as a Tool for Thinking

J's notation is not just compact—it’s a **tool for thought**. Originally envisioned by Kenneth Iverson as a replacement for pseudocode and math notation, J can serve as **executable whiteboard notation** in AI research:

* Define models, loss functions, data pipelines, and numerical experiments in a form **both symbolic and executable**.
* Embed J expressions directly in papers, documents, or lab notebooks—like literate programming for AI research.

Imagine replacing verbose algorithm boxes with single-line J expressions that **encode, clarify, and execute** ideas.

---

### 🛠️ Designed for Composability and Reuse

J is compositional by nature. Its **verbs and adverbs** form reusable, chainable building blocks—ideal for defining model architectures, transformations, and training loops without boilerplate.

Compare this to imperative frameworks where logic becomes entangled in mutable state, callbacks, and configuration files. J is **pure, expressive, and programmable at the meta level**.

---

### 🚧 Current Roadblocks (and Opportunity)

Yes, J needs:

* **Bindings to TensorFlow**, **MPI**, and **CUDA**
* A modern **J interpreter built for HPC** (we’re working on that)
* Better interop with Python and C++ (FFI + Jupyter kernel, perhaps)

But these are solvable. And once solved, J can become a **powerful AI DSL** that combines:

* The **conciseness of APL**
* The **semantic precision of math**
* The **performance of C++/CUDA**
* And the **expressiveness of functional programming**

---

### 🔚 Final Line

> **"J is not a language you write programs in. J is the language you think your programs in."**

In an age of AI, massive datasets, and distributed compute, the ability to **think clearly, write concisely, and execute efficiently** is more important than ever. J unifies these dimensions.

---

### Setup

Ensure you have CMake and a C++20 compatible compiler.

use git to initialize submodules with tensorflow

build tensorflow using bazel

### From the j_interpreter directory:
Bash
~~~
mkdir build
cd build
cmake ..
cmake --build .  # Or make, or your generator's build command
~~~
### To run tests (after building):
Bash
~~~
cd build
ctest # Or directly run ./tests/j_interpreter_tests
~~~
### To run the app:
Bash
~~~
cd build
./app/j_repl
~~~