import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.optim as optim

from simpletuner.helpers.training.custom_schedule import (
    CosineDecayPeak,
    enforce_zero_terminal_snr,
    get_polynomial_decay_schedule_with_warmup,
    patch_scheduler_betas,
    segmented_timestep_selection,
)


class TestPolynomialDecayWithWarmup(unittest.TestCase):
    def test_polynomial_decay_schedule_with_warmup(self):
        optimizer = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=0.1)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

        # Test warmup
        ranges = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
        first_lr = round(scheduler.get_last_lr()[0], 2)
        for step in range(len(ranges)):
            last_lr = round(scheduler.get_last_lr()[0], 2)
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(last_lr, ranges[-1], places=3)

        # Test decay
        for step in range(len(ranges), 100):
            optimizer.step()
            scheduler.step()
        # Implement your decay formula here to check
        expected_lr = 1e-7
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_lr, places=4)

    def test_enforce_zero_terminal_snr(self):
        betas = torch.tensor([0.9, 0.8, 0.7])
        new_betas = enforce_zero_terminal_snr(betas)
        final_beta = new_betas[-1]
        self.assertEqual(final_beta, 1.0)

    def test_patch_scheduler_betas(self):
        # Create a dummy scheduler with betas attribute
        class DummyScheduler:
            def __init__(self):
                self.betas = torch.tensor([0.9, 0.8, 0.7])

        scheduler = DummyScheduler()
        # Check value before.
        final_beta = scheduler.betas[-1]
        self.assertEqual(final_beta, 0.7)

        patch_scheduler_betas(scheduler)

        final_beta = scheduler.betas[-1]
        self.assertEqual(final_beta, 1.0)

    def test_inverted_schedule(self):
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args",
            return_value=MagicMock(
                refiner_training=True,
                refiner_training_invert_schedule=True,
                refiner_training_strength=0.35,
            ),
        ):
            weights = torch.ones(1000)  # Uniform weights
            selected_timesteps = segmented_timestep_selection(
                1000,
                10,
                weights,
                config=MagicMock(
                    refiner_training=True,
                    refiner_training_invert_schedule=True,
                    refiner_training_strength=0.35,
                ),
                use_refiner_range=False,
            )
            self.assertTrue(
                all(350 <= t <= 999 for t in selected_timesteps),
                f"Selected timesteps: {selected_timesteps}",
            )

    def test_normal_schedule(self):
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args",
            return_value=MagicMock(
                refiner_training=True,
                refiner_training_invert_schedule=False,
                refiner_training_strength=0.35,
            ),
        ):
            weights = torch.ones(1000)  # Uniform weights
            selected_timesteps = segmented_timestep_selection(
                1000,
                10,
                weights,
                use_refiner_range=False,
                config=MagicMock(
                    refiner_training=True,
                    refiner_training_invert_schedule=False,
                    refiner_training_strength=0.35,
                ),
            )
            self.assertTrue(all(0 <= t < 350 for t in selected_timesteps))

    def test_cosine_decay_peak_no_warmup_smoke(self):
        optimizer = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=1e-3)
        scheduler = CosineDecayPeak(
            optimizer=optimizer,
            T_0=5,
            max_steps=20,
            lr_end=1e-4,
            lr_power=1.0,
            warmup_steps=0,
            lr_floor=4e-4,
        )

        lrs = []
        peaks = []
        floors = []
        for step in range(0, 21):
            scheduler.step(step)
            lrs.append(scheduler.get_last_lr()[0])
            peaks.append(scheduler.get_peak_lr(step, base_lr=1e-3))
            floors.append(scheduler.get_floor_lr(step))

        tolerance = 1e-9
        self.assertTrue(
            all(
                (1e-4 - tolerance) <= floors[i] <= (lrs[i] + tolerance) <= (peaks[i] + tolerance)
                for i in range(len(lrs))
            )
        )
        self.assertAlmostEqual(floors[0], 4e-4, places=7)
        self.assertAlmostEqual(lrs[5], floors[5], places=7)
        self.assertAlmostEqual(floors[-1], 1e-4, places=7)

        self.assertTrue(all(peaks[i] >= peaks[i + 1] for i in range(len(peaks) - 1)))
        self.assertTrue(all(floors[i] >= floors[i + 1] for i in range(len(floors) - 1)))
        self.assertAlmostEqual(peaks[0], 1e-3, places=7)
        self.assertAlmostEqual(peaks[-1], 1e-4, places=7)

    def test_cosine_decay_peak_warmup_and_resume_smoke(self):
        optimizer = optim.SGD(
            [
                {"params": [torch.randn(2, 2, requires_grad=True)], "lr": 1e-3},
                {"params": [torch.randn(2, 2, requires_grad=True)], "lr": 5e-4},
            ]
        )
        scheduler = CosineDecayPeak(
            optimizer=optimizer,
            T_0=4,
            max_steps=16,
            lr_end=1e-4,
            lr_power=1.0,
            warmup_steps=8,
            warmup_start_lr=1e-4,
            lr_floor=4e-4,
            lr_floor_start=1e-4,
        )

        warmup_peaks = [scheduler.get_peak_lr(step, base_lr=1e-3) for step in range(0, 9)]
        warmup_floors = [scheduler.get_floor_lr(step) for step in range(0, 9)]
        self.assertTrue(all(warmup_peaks[i] <= warmup_peaks[i + 1] for i in range(len(warmup_peaks) - 1)))
        self.assertTrue(all(warmup_floors[i] <= warmup_floors[i + 1] for i in range(len(warmup_floors) - 1)))
        self.assertAlmostEqual(warmup_peaks[-1], 1e-3, places=7)
        self.assertAlmostEqual(warmup_floors[-1], 4e-4, places=7)
        self.assertAlmostEqual(scheduler.get_peak_lr(16, base_lr=1e-3), 1e-4, places=7)
        self.assertAlmostEqual(scheduler.get_floor_lr(16), 1e-4, places=7)
        scheduler.step(4)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], scheduler.get_floor_lr(4), places=7)

        total_steps = 12
        full_lrs = []
        for step in range(total_steps):
            scheduler.step(step)
            full_lrs.append(tuple(scheduler.get_last_lr()))

        resume_step = 5
        optimizer_resumed = optim.SGD(
            [
                {"params": [torch.randn(2, 2, requires_grad=True)], "lr": 1e-3},
                {"params": [torch.randn(2, 2, requires_grad=True)], "lr": 5e-4},
            ]
        )
        resumed_scheduler = CosineDecayPeak(
            optimizer=optimizer_resumed,
            T_0=4,
            max_steps=16,
            lr_end=1e-4,
            lr_power=1.0,
            warmup_steps=8,
            warmup_start_lr=1e-4,
            lr_floor=4e-4,
            lr_floor_start=1e-4,
        )
        resumed_scheduler.last_step = resume_step
        resumed_scheduler.last_epoch = resume_step

        resumed_lrs = []
        for _ in range(total_steps - (resume_step + 1)):
            resumed_scheduler.step()
            resumed_lrs.append(tuple(resumed_scheduler.get_last_lr()))

        self.assertEqual(full_lrs[resume_step + 1 :], resumed_lrs)

    def test_cosine_decay_peak_eta_min_alias_behaves_as_floor_max(self):
        optimizer = optim.SGD([torch.randn(2, 2, requires_grad=True)], lr=1e-3)
        scheduler = CosineDecayPeak(
            optimizer=optimizer,
            T_0=4,
            max_steps=16,
            lr_end=1e-4,
            lr_power=1.0,
            warmup_steps=8,
            warmup_start_lr=1e-4,
            eta_min=4e-4,
        )

        self.assertAlmostEqual(scheduler.get_floor_lr(0), 1e-4, places=7)
        self.assertAlmostEqual(scheduler.get_floor_lr(8), 4e-4, places=7)
        self.assertAlmostEqual(scheduler.get_floor_lr(16), 1e-4, places=7)


if __name__ == "__main__":
    unittest.main()
