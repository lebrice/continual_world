import pytest
import tensorflow as tf


# IDEA: To help with debugging, prevent the tf.function decorators from working when testing.
@pytest.fixture(autouse=True)
def disable_tf_function_wrappers():
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(False)
