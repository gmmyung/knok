import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("macos_linker", Path(__file__).with_name("macos-linker.py"))
linker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linker)


class MacosLinkerTests(unittest.TestCase):
    def test_executables_keep_llvm_symbols_private(self):
        self.assertEqual(linker.linker_args(["-o", "build-script-build"]),
                         ["-o", "build-script-build", "-Wl,-no_exported_symbols"])

    def test_shared_libraries_preserve_rust_entry_points(self):
        for kind in ("-dynamiclib", "-shared", "-bundle"):
            with self.subTest(kind=kind):
                arguments = [kind, "-Wl,-exported_symbols_list,exports.txt"]
                self.assertEqual(linker.linker_args(arguments), arguments)

    def test_lto_optimization_levels_and_cpu_reach_ld64(self):
        for level in range(4):
            with self.subTest(level=level):
                arguments = [f"-Wl,-dead_strip,-plugin-opt=O{level},-plugin-opt=mcpu=apple-m1"]
                self.assertEqual(linker.linker_args(arguments), [
                    f"-Wl,-dead_strip,--lto-O{level},--lto-CGO{level},-mcpu,apple-m1",
                    "-Wl,-no_exported_symbols",
                ])

    def test_other_arguments_are_preserved(self):
        arguments = ["file with spaces.o", "-Wl,-rpath,/tmp/path", "-fuse-ld=lld"]
        self.assertEqual(linker.linker_args(arguments), arguments + ["-Wl,-no_exported_symbols"])


if __name__ == "__main__":
    unittest.main()
