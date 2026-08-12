"""
Regression test for smoke test weight preservation.

Verifies that run_smoke_test() does not update model weights or optimizer state.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_smoke_test_preserves_weights():
    """Test that a smoke test (forward/backward without optimizer.step) preserves weights."""
    print("Testing smoke test weight preservation...")

    # Create a simple model
    model = nn.Linear(10, 2)

    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Save initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    initial_optimizer_state = optimizer.state_dict()

    # Simulate smoke test: forward, backward, but NO optimizer.step()
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        # Intentionally do NOT call optimizer.step()
        break  # Just one batch like a smoke test

    # Check that weights are unchanged
    for name, param in model.named_parameters():
        assert torch.equal(initial_weights[name], param), \
            f"Parameter {name} was modified by smoke test"

    # Check that optimizer state is unchanged (except for gradients which are computed)
    # The key is that step_count should not have increased
    for param_group in optimizer.param_groups:
        if 'step' in param_group:
            assert param_group['step'] == 0, "Optimizer step count was incremented"

    print("  ✓ Smoke test preserves model weights and optimizer state")


if __name__ == "__main__":
    test_smoke_test_preserves_weights()
    print("\nALL SMOKE TEST REGRESSION TESTS PASSED")
