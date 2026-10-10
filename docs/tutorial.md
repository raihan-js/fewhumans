# 10-minute tutorial (HelpSteer2 correctness, public data)

```bash
pip install -e ".[dev]"
# 1. Make a session from 200 HelpSteer2 validation items (see
#    study/run_judges/tasks.py::load_helpsteer2 for the slice code).
# 2. Run two local judges (Ollama, $0):
python study/run_judges/run_local.py --task helpsteer2-correctness \
  --judge local-qwen35-9b --n 200
# 3. In Python:
import fewhumans as fh
from fewhumans.core.session import Session
# ... create Session with task Mean(of=correctness), panel of judges,
# routing Uniform(budget=200); s.run_panel(); label s.queue(20) ...
est = s.estimate("E3-ppi-plus")   # value, interval, n_human, method
print(est)
s.should_stop(target_halfwidth=0.10)
s.report("report.html")           # opens the judge report card
```

What you should see: the PPI++ interval is narrower than human-only at the
same budget, both cover the full-human mean, and the judge-only number sits
outside its own naive interval roughly as often as our study found (18/21).
