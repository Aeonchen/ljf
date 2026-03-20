import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from src.shared.plotting import save_figure, safe_close


class PlottingTests(unittest.TestCase):
    def test_save_figure_without_gui(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            output = save_figure(fig, Path(tmpdir) / 'plot.png')
            safe_close(fig)
            self.assertTrue(Path(output).exists())


if __name__ == '__main__':
    unittest.main()
