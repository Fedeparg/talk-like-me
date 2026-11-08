# Privacy

[Back to the README](../README.md)

This project trains on private conversations, mine and other people's. These are the rules it follows, and how each one is enforced.

## What never leaves my home network

- The WhatsApp database, the processed messages, the datasets, the generated replies, the trained adapters and the fine-tuned classifiers. `data/`, `adapters/`, `models/`, `runs/` and `outputs/` are git-ignored, and so is every `.jsonl`, `.sqlite` and `.txt`.
- They live on my laptop and, for the later experiments, on a GPU server in my home, reached over my local network ([decision 24](../DECISIONS.md)). Nothing is uploaded to a cloud service.
- The adapters are treated like the data. The memorization study shows why: an adapter trained on my replies ranks an invented secret first among 1,001 candidates after it appears in only 4 training replies, and a 1.5B model writes half of such codes out in full after 16 repetitions (8 at a later step) ([results 2](results.md#2-does-it-memorize-private-text-decisions-13-23-27)). Anyone with the weights could try the same on real details.

## What the model sees

- **Placeholders instead of names, links, emails and phone numbers.** Links, emails, Spanish phone numbers and the names on a hand-curated contact list are replaced with `<URL>`, `<EMAIL>`, `<PHONE>` and `<NAME>` before the datasets are built ([data.md](data.md#cleaning-replace-never-delete)), so the model is trained on the placeholders. People outside the list, foreign numbers and addresses are not caught, which is another reason the adapters stay private. The name list was checked with numbers: one bad entry produced 455 false replacements and was removed.
- **Only one-to-one chats**, never groups, and chats can be excluded by id (bots, businesses, anyone who would not want it).
- **Only my replies are learned.** Training uses `--mask-prompt`: the other person's messages are context, and the loss never asks the model to produce them.

## Who read the messages

- Training and evaluation run on my own machines. The only network access is downloading the base models from Hugging Face; no data is uploaded.
- I used an AI coding assistant for the plumbing, under one rule: the commands it ran on the data print counts, dates and metrics, not message text or names. That rule is how I worked; it is not something the code enforces. How I split the work with it is in [decision 10](../DECISIONS.md).  
- `report.py --show-examples` and `run.py --list-chats` print text or contact names. They are meant for my own terminal only.

## What is published

- Code, the decision log, aggregated numbers (`experiments/results/*.json` holds losses, counts, ranks and session ids, no text), figures made from those numbers, and the notebooks with their outputs, which print numbers and the structure of examples (who spoke and how long), not the other person's words. The one exception is `Canaries.ipynb`, which shows one short reply of mine, with nothing personal in it, next to the planted invented secret.
- Examples of messages in the docs are invented. Words that describe my own style (for example how often I write "xd") are mine to share. Nothing that describes another person is published.
