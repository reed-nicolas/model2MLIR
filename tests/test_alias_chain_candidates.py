"""A lineage-sharing lowering chain binds its final value; independent values stay ambiguous."""
from m2m.ir.import_fx import chain_result_candidates


class _Owner:
    def __init__(self, operands):
        self.operands = operands


class _Value:
    """Hashable stand-in for an SSA value: identity hashing, like xDSL values."""

    def __init__(self, operands):
        self.owner = _Owner(operands)


def _value(*operands):
    return _Value(tuple(operands))


def test_chain_keeps_only_the_unconsumed_result():
    source = _value()
    sigmoid = _value(source)
    product = _value(source, sigmoid)
    assert chain_result_candidates({sigmoid, product}) == {product}


def test_longer_chain_keeps_its_last_value():
    first = _value()
    second = _value(first)
    third = _value(second)
    assert chain_result_candidates({first, second, third}) == {third}


def test_independent_values_both_survive_for_the_caller_to_refuse():
    left, right = _value(), _value()
    assert chain_result_candidates({left, right}) == {left, right}


def test_single_or_empty_candidates_are_unchanged():
    only = _value()
    assert chain_result_candidates({only}) == {only}
    assert chain_result_candidates(set()) == set()
