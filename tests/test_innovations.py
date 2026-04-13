from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.envs.amazons_env import AmazonsConfig, MiniAmazonsEnv
from src.train.innovation_utils import prune_actions_by_score, save_case_record, save_run_metadata


class InnovationTests(unittest.TestCase):
    def test_reward_shaping_changes_step_reward(self):
        env_plain = MiniAmazonsEnv(AmazonsConfig(size=6, max_turns=50))
        env_shape = MiniAmazonsEnv(
            AmazonsConfig(
                size=6,
                max_turns=50,
                reward_mobility_weight=0.05,
                reward_center_weight=0.02,
            )
        )
        s0 = env_plain.reset()
        s1 = env_shape.reset()
        self.assertEqual(s0, s1)
        a = env_plain.legal_actions(0)[0]
        _, r_plain, _, _ = env_plain.step(a)
        _, r_shape, _, _ = env_shape.step(a)
        self.assertNotEqual(r_plain[0], r_shape[0])

    def test_pruning_reduces_action_space(self):
        env = MiniAmazonsEnv(AmazonsConfig(size=6, max_turns=50))
        env.reset()
        legal = env.legal_actions(0)
        pruned = prune_actions_by_score(env, 0, legal, top_k=8, keep_ratio=1.0)
        self.assertGreater(len(legal), 8)
        self.assertEqual(len(pruned), 8)

    def test_trace_files_generated(self):
        with tempfile.TemporaryDirectory() as td:
            meta_path = str(Path(td) / "meta.json")
            case_path = str(Path(td) / "case.json")
            run_id = save_run_metadata(meta_path, "unit_test", {"x": 1}, extra={"ok": True})
            case_id = save_case_record(case_path, {"who": "test"}, [{"turn": 1}])
            self.assertTrue(run_id)
            self.assertTrue(case_id)
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            case = json.loads(Path(case_path).read_text(encoding="utf-8"))
            self.assertEqual(meta["run_id"], run_id)
            self.assertEqual(case["case_id"], case_id)


if __name__ == "__main__":
    unittest.main()
