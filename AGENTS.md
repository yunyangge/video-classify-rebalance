# Working rules

- Keep one input collection and one selected output list.
- Keep metadata, model weights, business configurations and run outputs outside Git.
- Encode the full joined tag text once per video with Qwen3-Embedding-4B at 2560 dimensions.
- Apply density limits to totals and category aggregates, allowing individual high-density videos.
- Keep direct category endpoints, source metadata and explicit target-hour denominators.
- Run `python -m unittest discover -s tests -v` after code changes.
- Keep user documentation concise and distinguish generated tests from real model measurements.
