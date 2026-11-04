"""Where Linux finds CUDA's libraries: what the extension adds to the loader's path before it starts again."""

import sys
import tempfile
import unittest
from pathlib import Path

from router import libpaths


class LibpathsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prefix = Path(self.tmp.name)
        site = self.prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        for name in ("cu13", "cudnn"):
            (site / "nvidia" / name / "lib").mkdir(parents=True)
        self.libs = [str(site / "nvidia" / n / "lib") for n in ("cu13", "cudnn")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_pips_cuda_libraries_go_first(self):
        env = libpaths.wanted(False, {"LD_LIBRARY_PATH": "/opt/x"}, self.prefix)
        self.assertEqual(env, {"LD_LIBRARY_PATH": ":".join(self.libs + ["/opt/x"])})

    def test_nothing_missing_is_nothing_to_add(self):
        self.assertEqual(libpaths.wanted(False, {"LD_LIBRARY_PATH": ":".join(self.libs)}, self.prefix), {})

    def test_no_cuda_libraries_no_restart(self):
        self.assertEqual(libpaths.wanted(False, {}, self.prefix / "elsewhere"), {})

    def test_nixos_adds_the_drivers_folder_when_it_has_one(self):
        env = libpaths.wanted(True, {}, self.prefix)
        have = Path("/run/opengl-driver/lib").is_dir()
        self.assertEqual(env["LD_LIBRARY_PATH"].split(":")[0] == "/run/opengl-driver/lib", have)

    def test_it_starts_again_only_once(self):
        self.assertTrue(libpaths.DONE)


if __name__ == "__main__":
    unittest.main()
