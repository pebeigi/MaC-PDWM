"""Does a restored branch reproduce the rollout it was saved from?

Everything causal in this project - paired do(u) data, the branch oracle, the
counterfactual evaluation - assumes that saving a state, rolling forward, and
reloading returns the simulator to the identical state. If it does not, those
comparisons are between different scenes and mean nothing.
"""
import os
import tempfile
import unittest

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv


class BranchFidelityTests(unittest.TestCase):
    def test_restored_branch_reproduces_rollout(self):
        cfg = EnvConfig(scenario="cross", seed=0, horizon=80)
        env = SumoPlanningEnv(cfg, label="branch_fidelity", continuous=True)
        try:
            env.reset(seed=4321)
            for _ in range(25):
                env.step(np.array([0.0], dtype=np.float32))

            actions = [1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, -3.0]
            with tempfile.TemporaryDirectory(prefix="mac_fid_") as tmp:
                path = os.path.join(tmp, "state.xml")
                state = env.save_branch_state(path)

                def replay():
                    rewards, positions = [], []
                    for accel in actions:
                        _, reward, _, _ = env.step(
                            np.array([accel], dtype=np.float32))
                        rewards.append(float(reward[0]))
                        snap = env._last_snapshot.get(env.egos[0].veh_id)
                        positions.append(
                            (snap["x"], snap["y"], snap["speed"])
                            if snap else (np.nan,) * 3)
                    return np.array(rewards), np.array(positions)

                first_rewards, first_positions = replay()
                env.load_branch_state(path, state)
                second_rewards, second_positions = replay()

            # Speeds are commanded, so they must reproduce exactly. Position
            # carries a residual few-centimetre offset that appears once a
            # background vehicle is inserted; it is far below the metre-scale
            # effects being measured, but the bound keeps it from growing.
            np.testing.assert_allclose(
                second_positions[:, 2], first_positions[:, 2], atol=1e-9,
                err_msg="ego speed diverges after branch restore")
            offset = np.abs(second_positions[:, :2] - first_positions[:, :2]).max()
            self.assertLess(
                offset, 0.25,
                f"ego position diverges by {offset:.3f} m after branch restore")
            np.testing.assert_allclose(
                second_rewards, first_rewards, atol=1e-3,
                err_msg="reward stream diverges after branch restore")
        finally:
            env.close()


    def test_ego_still_moves_after_a_branch_reaches_arrival(self):
        """The failure that made every matched-branch comparison meaningless.

        Scoring several candidates from one state runs branches to completion.
        A simulator rewound past an arrival refuses to re-insert that vehicle,
        so the ego freezes at its departure point while time keeps advancing.
        """
        cfg = EnvConfig(scenario="cross", seed=0)
        env = SumoPlanningEnv(cfg, label="branch_arrival", continuous=True)
        try:
            env.reset(seed=200_000)
            with tempfile.TemporaryDirectory(prefix="mac_arr_") as tmp:
                path = os.path.join(tmp, "state.xml")
                state = env.save_branch_state(path)
                for accel in (3.0, -4.0, 3.0):
                    env.load_branch_state(path, state)
                    for _ in range(60):
                        _, _, dones, _ = env.step(
                            np.array([accel], dtype=np.float32))
                        if bool(dones[0]):
                            break
                env.load_branch_state(path, state)
                ego = env.egos[0].veh_id
                start = env._last_snapshot[ego]["y"]
                for _ in range(10):
                    env.step(np.array([2.0], dtype=np.float32))
                moved = abs(env._last_snapshot[ego]["y"] - start)
            self.assertGreater(
                moved, 5.0,
                "ego is frozen after branches ran to arrival; matched-branch "
                "comparisons would measure the freeze, not the intervention")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
