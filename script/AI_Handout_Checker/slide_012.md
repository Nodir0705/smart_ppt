To close, here is a peek under the hood for the curious, an appendix with the technical details of how the checker works inside.
First, two local models share the job: a Nanonets vision model reads the handwriting, and a small Qwen model does the grading.
Both are quantized to four bits and loaded one at a time, which is how they squeeze into eight gigabytes of graphics memory.
Next, circled answers are found with plain computer vision and zero false positives, while Tesseract word boxes, column splits and exercise regions decide exactly where each tick and cross lands on the page.
Then Python recomputes every score so the grading is deterministic, a teacher's answer key can override any item, and everything is cached on disk so repeat runs skip loading the models entirely.
Finally, the whole stack, PyTorch and Transformers, runs completely offline, so your students' work never leaves the machine, and that brings us to the end of the talk, thank you.
