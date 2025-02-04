from tests.test_utils import load_one_hot, load_preprocess_example


def test_one_hot():
    one_hot_data = load_one_hot()
    assert one_hot_data.shape == (49, 5)
    assert one_hot_data.sum() == 49
    assert one_hot_data[0][0] == 1
    assert one_hot_data[20][1] == 1
    assert one_hot_data[-1][2] == 0


def preprocess_data():
    new_data = load_preprocess_example()
    shape_out = new_data.node_direction_expansion.shape
    assert shape_out == (450, 16), f"Expected shape (450, 16), got {shape_out}"
