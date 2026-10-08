"""CPU-only tests for the focused LIBERO-PRO experiment registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_vla.benchmarks.libero_pro import (
    LIBERO_PRO_MAIN_TASK_IDS,
    build_libero_pro_haa_index,
    libero_pro_qualification_matrix,
    validate_libero_pro_root,
)


class LiberoProConfigurationTest(unittest.TestCase):
    def test_matrix_is_focused_on_pre_registered_tasks(self) -> None:
        matrix = libero_pro_qualification_matrix()

        self.assertEqual(LIBERO_PRO_MAIN_TASK_IDS, (3, 8, 9))
        self.assertEqual(
            {cell.suite for cell in matrix},
            {"libero_10", "libero_10_swap", "libero_10_object", "libero_10_task"},
        )
        self.assertTrue(all(set(cell.task_ids) <= {3, 8, 9} for cell in matrix))

    def test_haa_index_retrieves_articulation_and_object_cards(self) -> None:
        records = build_libero_pro_haa_index().retrieve(
            task_instruction="put the mug in the microwave and close it",
            failure_type="collision",
            limit=3,
        )

        ids = {record["experience_id"] for record in records}
        self.assertIn("libero-mug-upright-handle", ids)
        self.assertIn("libero-microwave-articulation", ids)

    def test_validation_rejects_original_libero_for_pro_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = (
                Path(directory)
                / "libero"
                / "libero"
                / "benchmark"
                / "__init__.py"
            )
            registry.parent.mkdir(parents=True)
            registry.write_text('self.name = "libero_10"\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "official LIBERO-PRO"):
                validate_libero_pro_root(directory, "libero_10_swap")

    def test_validation_requires_downloaded_suite_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = (
                Path(directory)
                / "libero"
                / "libero"
                / "benchmark"
                / "__init__.py"
            )
            registry.parent.mkdir(parents=True)
            registry.write_text('self.name = "libero_10_swap"\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "requires 10 official"):
                validate_libero_pro_root(directory, "libero_10_swap")


if __name__ == "__main__":
    unittest.main()
