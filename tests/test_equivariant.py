import torch

from galois_ml.equivariant import (
    CyclicEquivariantLinear,
    GaloisInvariantNet,
    PlainMLP,
)


def test_cyclic_layer_is_equivariant():
    torch.manual_seed(0)
    n, c_in, c_out, batch = 8, 3, 5, 4
    layer = CyclicEquivariantLinear(c_in, c_out, n)
    x = torch.randn(batch, c_in, n)
    y = layer(x)
    for shift in range(1, n):
        y_shifted = layer(torch.roll(x, shifts=shift, dims=-1))
        assert torch.allclose(
            y_shifted, torch.roll(y, shifts=shift, dims=-1), atol=1e-5
        )


def test_invariant_net_is_frobenius_invariant():
    torch.manual_seed(0)
    p, n = 5, 4
    net = GaloisInvariantNet(p, n, n_classes=2)
    coords = torch.randint(0, p, (16, n))
    out = net(coords)
    for shift in range(1, n):
        out_shifted = net(torch.roll(coords, shifts=shift, dims=-1))
        assert torch.allclose(out_shifted, out, atol=1e-5)


def test_plain_mlp_is_not_invariant():
    # sanity check that the baseline actually lacks the symmetry
    torch.manual_seed(0)
    p, n = 5, 4
    net = PlainMLP(p, n, n_classes=2)
    coords = torch.randint(0, p, (16, n))
    out = net(coords)
    out_shifted = net(torch.roll(coords, shifts=1, dims=-1))
    assert not torch.allclose(out_shifted, out, atol=1e-5)


def test_shapes():
    p, n = 3, 6
    net = GaloisInvariantNet(p, n, n_classes=4, hidden_channels=8, depth=2)
    coords = torch.randint(0, p, (7, n))
    assert net(coords).shape == (7, 4)


def test_group_equivariant_layer_all_groups():
    import torch as _torch

    from galois_ml.equivariant import GroupEquivariantLinear
    from galois_ml.group_algebra import REALIZATIONS

    _torch.manual_seed(0)
    for name in ("S3", "D4", "Q8", "C4xC2"):
        group = REALIZATIONS[name]
        layer = GroupEquivariantLinear(group, 2, 3)
        x = _torch.randn(5, 2, group.order)
        y = layer(x)
        for g in range(group.order):
            perm = _torch.from_numpy(group.left_translation(g)).long()
            y_translated = layer(x[:, :, perm])
            assert _torch.allclose(y_translated, y[:, :, perm], atol=1e-5), (
                f"{name}: not equivariant under element {g}"
            )


def test_group_invariant_net_is_invariant():
    import torch as _torch

    from galois_ml.equivariant import GroupInvariantNet
    from galois_ml.group_algebra import REALIZATIONS

    _torch.manual_seed(0)
    group = REALIZATIONS["Q8"]
    net = GroupInvariantNet(group, n_classes=3, hidden_channels=8, depth=2)
    x = _torch.randn(4, group.order)
    out = net(x)
    for g in range(group.order):
        perm = _torch.from_numpy(group.left_translation(g)).long()
        assert _torch.allclose(net(x[:, perm]), out, atol=1e-5)


def test_cyclic_layer_agrees_with_group_layer():
    """CyclicEquivariantLinear must be the G = C_n special case of
    GroupEquivariantLinear."""
    import torch as _torch

    from galois_ml.equivariant import GroupEquivariantLinear
    from galois_ml.group_algebra import cyclic

    _torch.manual_seed(0)
    n = 6
    group_layer = GroupEquivariantLinear(cyclic(n), 2, 3)
    cyc = CyclicEquivariantLinear(2, 3, n)
    with _torch.no_grad():
        cyc.weight.copy_(group_layer.weight)
        cyc.bias.copy_(group_layer.bias)
    x = _torch.randn(4, 2, n)
    assert _torch.allclose(cyc(x), group_layer(x), atol=1e-5)
