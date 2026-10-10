"""Example scripts stay working (tiny runs)."""
from examples.pairwise_winrate import main as pairwise_main
from examples.quickstart_accuracy import main as quickstart_main


def test_quickstart_accuracy(tmp_path):
    out = quickstart_main(n_items=200, n_human=20, seed=0, outdir=str(tmp_path / "q"))
    assert abs(out["e3"]["value"] - 0.72) < 0.15
    assert (tmp_path / "q" / "report.html").exists()


def test_pairwise_winrate(tmp_path):
    out = pairwise_main(n_items=200, n_human=30, seed=0, outdir=str(tmp_path / "p"))
    assert abs(out["e5"]["value"] - 0.58) < 0.2
    assert (tmp_path / "p" / "report.html").exists()
