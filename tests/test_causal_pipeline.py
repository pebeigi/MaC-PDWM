import numpy as np
import torch
import unittest

from mac.data.normalize import decode_samples, encode_target, normalize_inputs
from mac.data.scene import synthetic_plan
from mac.envs.social_drivers import SocialDriverManager
from mac.models.diffusion_world_model import DiffusionWorldModel


class CausalPipelineTests(unittest.TestCase):
    def test_residual_encode_decode_round_trip(self):
        history = torch.zeros(2, 5, 3, 5)
        history[:, -1, 1:, :2] = torch.tensor(
            [[[3.0, -2.0], [1.0, 4.0]]])
        history[:, -1, 1:, 2:4] = torch.tensor(
            [[[5.0, 0.0], [0.0, -2.0]]])
        future = torch.zeros(2, 4, 2, 3)
        future[..., :2] = torch.randn(2, 4, 2, 2)
        future[..., 2] = 1.0

        encoded = encode_target(history, future)
        decoded = decode_samples(encoded[:, None], history)[:, 0]
        expected = history[:, -1, 1:, :2][:, None] + future[..., :2]
        torch.testing.assert_close(decoded, expected)

    def test_normalization_does_not_change_presence_mask(self):
        history = torch.randn(2, 5, 4, 5)
        history[..., 4] = torch.randint(0, 2, history[..., 4].shape)
        plan = torch.randn(2, 10, 3)
        normalized, _ = normalize_inputs(history, plan)
        torch.testing.assert_close(normalized[..., 4], history[..., 4])

    def test_common_noise_aligns_counterfactual_latents(self):
        model = DiffusionWorldModel(
            history_len=2, future_len=3, n_neighbors=1, plan_len=3,
            n_steps=4, hidden=16, context_dim=8)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        history = torch.zeros(2, 2, 2, 5)
        plans = torch.stack([
            torch.from_numpy(synthetic_plan(5.0, -2.0, 3, 0.4, 10.0)),
            torch.from_numpy(synthetic_plan(5.0, 2.0, 3, 0.4, 10.0)),
        ])
        torch.manual_seed(0)
        samples = model.sample(
            history, plans, n_samples=3, steps=2, eta=0.0,
            common_noise=True)
        torch.testing.assert_close(samples[0], samples[1])

    def test_approach_gated_decision_zone(self):
        # Post-conflict vehicle with negative signed distance is out of zone.
        manager = SocialDriverManager(
            (0.0, -20.0), np.random.default_rng(0),
            decision_distance=22.0,
            approach_edge_prefixes=("W_in", ":nW_", "circ_WS", ":nS_"))
        past = {"x": 10.0, "y": -20.0, "heading": 0.0, "edge": "circ_SE"}
        self.assertLess(manager._signed_distance(
            past["x"], past["y"], past["heading"]), 0)
        self.assertFalse(manager._in_decision_zone(past))

        # Euclidean-near opposite arc is excluded by the edge allowlist.
        wrong_arc = {
            "x": -10.0, "y": 10.0, "heading": -2.0, "edge": "circ_NW"}
        self.assertFalse(manager._in_decision_zone(wrong_arc))

        # South-feeder approach inside the ball is in zone.
        approach = {
            "x": -12.0, "y": -18.0, "heading": -0.6, "edge": "circ_WS"}
        self.assertGreater(manager._signed_distance(
            approach["x"], approach["y"], approach["heading"]), 0)
        self.assertTrue(manager._in_decision_zone(approach))

    def test_cross_requires_approaching(self):
        manager = SocialDriverManager(
            (0.0, 0.0), np.random.default_rng(0), decision_distance=35.0)
        past = {"x": 5.0, "y": 0.0, "heading": 0.0, "edge": "E_out"}
        self.assertFalse(manager._in_decision_zone(past))
        approach = {"x": -20.0, "y": 0.0, "heading": 0.0, "edge": "W_in"}
        self.assertTrue(manager._in_decision_zone(approach))

    def test_roundabout_slots_drop_wrong_arc(self):
        from mac.data.scene import extract_scene
        prefixes = ("W_in", ":nW_", "circ_WS", ":nS_")
        frames = [{
            "vehicles": {
                "ego": (0.0, -40.0, 0.0, 8.0, 8.0, 1.57, True),
                "nw": (-10.0, 10.0, 0.0, 0.0, 8.0, -2.0, False),
                "ws": (-12.0, -18.0, 0.0, 0.0, 8.0, -0.6, False),
            },
            "edges": {"ego": "S_in", "nw": "circ_NW", "ws": "circ_WS"},
        }]
        gated = extract_scene(
            frames, 0, "ego", 1, 5, conflict_point=(0.0, -20.0),
            approach_edge_prefixes=prefixes, decision_distance=22.0)
        self.assertEqual(gated["neighbor_ids"], ["ws"])
        ungated = extract_scene(
            frames, 0, "ego", 1, 5, conflict_point=(0.0, -20.0))
        self.assertIn("nw", ungated["neighbor_ids"])

    def test_far_queue_dropped_from_slots(self):
        from mac.data.scene import extract_scene
        frames = [{
            "vehicles": {
                "ego": (0.0, -40.0, 0.0, 8.0, 8.0, 1.57, True),
                "queue": (-80.0, -1.6, 8.0, 0.0, 8.0, 0.0, False),
                "ws": (-12.0, -18.0, 0.0, 0.0, 8.0, -0.6, False),
            },
            "edges": {"ego": "S_in", "queue": "W_in", "ws": "circ_WS"},
        }]
        prefixes = ("W_in", ":nW_", "circ_WS", ":nS_")
        scene = extract_scene(
            frames, 0, "ego", 1, 5, conflict_point=(0.0, -20.0),
            approach_edge_prefixes=prefixes, decision_distance=22.0)
        self.assertEqual(scene["neighbor_ids"][0], "ws")
        self.assertNotIn("queue", scene["neighbor_ids"][:1])

    def test_worst_partner_intent_pooling(self):
        from mac.models.belief import BeliefEncoder
        encoder = BeliefEncoder(
            None, "cpu", 0.4, 13.89, 1.0, 1.0, n_samples=1, sample_steps=1,
            mode="geometry", history_len=1, future_len=4, n_neighbors=2)
        intents = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        weights = np.array([0.5, 0.5], dtype=np.float32)
        p_yield, p_contest = encoder._pool_intents(intents, weights)
        self.assertAlmostEqual(p_yield, 0.2, places=5)
        self.assertAlmostEqual(p_contest, 0.8, places=5)

    def test_token_conditioned_model_masks_and_pairs(self):
        model = DiffusionWorldModel(
            history_len=2, future_len=3, n_neighbors=2, plan_len=3,
            n_steps=4, hidden=16, context_dim=8, token_conditioned=True)
        history = torch.randn(2, 2, 3, 5)
        history[..., 4] = 1.0
        history[:, :, 2, 4] = 0.0
        plan_a = torch.randn(2, 3, 3)
        plan_b = torch.randn(2, 3, 3)
        future_a = torch.randn(2, 3, 2, 3)
        future_b = torch.randn(2, 3, 2, 3)
        future_a[..., 2] = history[:, -1, 1:, 4][:, None]
        future_b[..., 2] = future_a[..., 2]
        loss, _ = model.loss(
            history, plan_a, future_a,
            trajectory_mask=torch.ones(2, dtype=torch.bool))
        delta = model.counterfactual_delta_loss(
            history, plan_a, future_a, history, plan_b, future_b)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(delta))

        types = torch.full((2, 2), -1)
        types[:, 0] = 0
        slot_w = model._labelled_slot_weight(types, future_a, 5.0)
        weights = slot_w.reshape(2, 3, 2, 2)
        self.assertTrue(torch.equal(weights[:, :, 0, :], torch.full((2, 3, 2), 5.0)))
        self.assertTrue(torch.equal(weights[:, :, 1, :], torch.full((2, 3, 2), 1.0)))

    def test_independent_denoiser_rejects_token_attention(self):
        with self.assertRaises(ValueError):
            DiffusionWorldModel(
                history_len=2, future_len=3, n_neighbors=2, plan_len=3,
                n_steps=4, hidden=16, context_dim=8,
                independent=True, token_conditioned=True)
        model = DiffusionWorldModel(
            history_len=2, future_len=3, n_neighbors=2, plan_len=3,
            n_steps=4, hidden=16, context_dim=8, independent=True)
        history = torch.zeros(1, 2, 3, 5)
        history[..., 4] = 1.0
        plan = torch.zeros(1, 3, 3)
        future = torch.zeros(1, 3, 2, 3)
        future[..., 2] = 1.0
        loss, _ = model.loss(history, plan, future)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
