# 10-minute tutorial (HelpSteer2 correctness, public data)

```bash
pip install fewhumans
# Fastest start (no data needed) — accuracy from judges + 50 human labels:
python examples/quickstart_accuracy.py      # (run from the repo checkout)
python examples/pairwise_winrate.py         # win rate + disagreement routing
```
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
