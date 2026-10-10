# Judge prompt templates (DRAFT — frozen + hashed before simulations, spec §7.2)
# Rules: written once, never tuned against human labels. Strict JSON output.
# {label_set} and {item} are filled per task. Version: v0-draft.

## ordinal.md
Rate the following on a {lo}..{hi} scale for {dimension}.
{item}
Reply with EXACTLY this JSON and nothing else: {"label": <integer>, "confidence": <0..1>}

## pairwise.md
Which response is better? A, B, or tie.
{item}
Reply with EXACTLY this JSON and nothing else: {"label": <"A"|"B"|"tie">, "confidence": <0..1>}

## binary.md
Classify the following as {neg} or {pos}.
{item}
Reply with EXACTLY this JSON and nothing else: {"label": <"LABEL">, "confidence": <0..1>}

## categorical.md
Assign exactly one of these labels: {label_set}.
{item}
Reply with EXACTLY this JSON and nothing else: {"label": <"LABEL">, "confidence": <0..1>}
