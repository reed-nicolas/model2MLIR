"""An exactly typed broadcast_tensors is proven to be the identity, and nothing else is."""
from m2m.capture.trace import _identity_broadcast


def _value(node_id, shape, stride, dtype="float32"):
    return {"id": f"{node_id}:v0", "kind": "tensor", "dtype": dtype, "storage_dtype": dtype, "compute_dtype": None,
            "shape": shape, "device": "cpu", "layout": "torch.strided", "stride": stride}


def _graph(result_shape, result_stride):
    producers = [{"id": name, "op": "call_function", "results": [_value(name, [1, 50, 32], [1600, 32, 1])]}
                 for name in ("a", "b")]
    results = [dict(_value("bt", result_shape, result_stride), id=f"bt:v{i}") for i in range(2)]
    node = {"id": "bt", "op": "call_function", "target": "aten.broadcast_tensors.default", "kwargs": {},
            "args": [[{"node_id": "a", "value_id": "a:v0"}, {"node_id": "b", "value_id": "b:v0"}]], "results": results}
    return {"nodes": [*producers, node]}, node


def test_same_abi_broadcast_is_an_eliminated_identity():
    source, node = _graph([1, 50, 32], [1600, 32, 1])
    proof = _identity_broadcast(source, node)
    assert proof["kind"] == "eliminated" and proof["source_ids"] == ["bt"]
    assert proof["proof"]["operand_value_ids"] == ["a:v0", "b:v0"]


def test_expanding_or_restrided_broadcast_is_not_an_identity():
    for shape, stride in (([2, 50, 32], [1600, 32, 1]), ([1, 50, 32], [0, 32, 1])):
        source, node = _graph(shape, stride)
        assert _identity_broadcast(source, node) is None
    source, node = _graph([1, 50, 32], [1600, 32, 1])
    assert _identity_broadcast(source, dict(node, target="aten.expand.default")) is None
