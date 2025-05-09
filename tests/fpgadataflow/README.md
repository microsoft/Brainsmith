# Writing tests with OpTest
Welcome! This folder contains unit tests for Brainsmith's custom operators. Though any test written in PyTest would suffice, it's reccommended you write your tests using **OpTest**, for a few reasons. But first...

# What is OpTest?
**OpTest** is an abstract class which uses PyTest's functionality to make writing tests for Brainsmith operators easier.

This README file serves as OpTest's documentation, for those trying to write tests with it. Let's get started!

# Reference

## Fixtures

Fixtures are functions that get used when parameters with the same name are used in test functions (or other fixtures). When this happends, PyTest automatically evaulates the fixture, and passes its return values to the test (or fixture) as the parameter with the same name.

These are useful for setting up the data your test will operate on. In OpTest, the model that contains your custom op will go through many stages (conversion to a hardware layer, specialisation, etc.). All stages the model goes through are represented as fixtures. This way any test can operate on any stage of the model, independently.

---

### `model`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| ModelWrapper  |

An abstract fixture that generates the QONNX ModelWrapper to be tested (when implemented). Each test MUST override this fixture, otherwise any PyTests will result in a NotImplementedError.

Helper functions such as create_model() and run_transforms() may be useful in reducing boilerplate when implementing this fixture.

---

### `model_hw`

| Dependencies  |
| ------------- |
| [`model`](#model)  |
| [`infer_hw_transform`](#infer_hw_transform)  |

| Returns  |
| ------------- |
| `ModelWrapper`  |

Converts all ONNX layers of a specific type to hardware layers, using a given inference function. If that function does not exist (i.e. infer_hw_transform == 'None'), then the model is directly passed through. All fixtures reliant on hardware inference should check if infer_hw_transform == 'None' before using this fixture.

---

### `model_specialised`

| Dependencies  |
| ------------- |
| [`model_hw`](#model_hw)  |
| [`target_fpga`](#target_fpga)  |
| [`output_dir`](#output_dir)  |

| Returns  |
| ------------- |
| `ModelWrapper`  |

A fixture that applies the `step_specialize_layers Builder` Step (using the `apply_builder_step()` helper function) to the 'model_hw' fixture, then returns the resulting ModelWrapper.

---

### `infer_hw_transform`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| `Transformation`  |

The transformation to infer a hardware layer from a standard ONNX layer. If this fixture returns 'None', OpTest assumes to skip hardware inference.

---

### `target_fpga`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| `str`  |

The fpga we're targeting for testing. Can be overridden by test classes. By default, returns `"xcv80-lsva4737-2MHP-e-S"`.

---

### `target_node`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| `int`  |

The index of the node in the model we're focusing on. Allows for multiple nodes to be present, with tests that only target a specific node. Defaults to the first node. Can be overridden. By default, returns `0`.

---

### `input_tensors`

| Dependencies  |
| ------------- |
| [`model`](#model)  |

| Returns  |
| ------------- |
| `dict`  |

Creates the tensor(s) passed to the model, to be used by the simulation during testing. By default, this fixture creates a tensor with random values, but can be overridden by tests to pass specific values.

---

### `save_intermediate_models`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| `bool`  |

If this fixture is overridden to return True, the 'auto_saver' fixture will automatically save the outputs of all 'model' fixtures to .onnx files. By default, returns `False`.

---

### `auto_saver`

> _This is an AUTOUSED fixture. This fixture does not need to be called for its code to be ran._

| Dependencies  |
| ------------- |
| [`request`](https://docs.pytest.org/en/7.1.x/reference/reference.html#std-fixture-request)  |
| [`save_intermediate_models`](#save_intermediate_models)  |
| [`output_dir`](#output_dir)  |

| Returns  |
| ------------- |
| None  |

Saves the output of each intermediate model step to a .onnx file.

If the 'save_intermediate_models' fixture evaluates to true, this fixture automatically scrapes all fixtures with names that start with 'model' (i.e. 'model', 'model_specialised') and saves their output to a .onnx file (if the fixtures return a ModelWrapper).

---

### `output_dir`

| Dependencies  |
| ------------- |
| None  |

| Returns  |
| ------------- |
| `str`  |

he directory we'll save the output of our tests to. By default, OpTest saves to a directory called "test_output", which is created in the same directory as the python file containing the test.

---

### `create_output_dir`

> _This is an AUTOUSED fixture. This fixture does not need to be called for its code to be ran._

| Dependencies  |
| ------------- |
| [`output_dir`](#output_dir)  |

| Returns  |
| ------------- |
| None  |

Automatically makes a test output directory if none exists. This fixture auto-runs before all test functions. If the directory exists, this fixture does nothing.

---

## Built-in Tests

When your test class inherits from OpTest, it also inherits these built-in tests. No extra code required! So far, there is only one built-in test available.

---

### `test_cycles`

| Dependencies  |
| ------------- |
| [`model_specialised`](#model_specialised)  |
| [`target_node`](#target_node)  |
|  `exec_mode` (A pytest parameter that evaluates to `"cppsim"` or `"rtlsim"`)  |

| Returns  |
| ------------- |
| None  |

Ensures the number of cycles the layer takes to run in rtlsim aligns with the expected number of cycles provided by the node's implementation of the node's `exp_cycles()` function.

---

## Helper functions

These functions make writing tests, and the fixtures they depend on, easier.

---

### `create_model(inputs, outputs, inits, nodes, opset, name)`

| parameters  |  Type  |
| ------------- | ------------- |
| `inputs`  |  `List`  |
| `outputs`  |  `List`  |
| `inits`  |  `List`  |
| `nodes`  |  `List`  |
| `opset`  |  `int` (default = 17)  |
| `name`  |  `List` (default = "OpTest_Graph")  |

| Returns  |
| ------------- |
| ModelWrapper  |

This wrapper function contains many onnx.helper functions used to create a graph, as well as many QONNX helper functions used to specify things like SIMD and in/out DataTypes. See example graphs for usage.

---

### `apply_transforms(model, transform_list, validate, input_tensors, tolerance)`

| parameters  |  Type  |
| ------------- | ------------- |
| `model`  |  `ModelWrapper`  |
| `transform_list`  |  `List[Transformation]` or `List[List[Transformation]]`  |
| `validate`  |  `bool` (default = False)  |
| `input_tensors`  |  `dict` (default = none)  |
| `tolerance`  |  `float`  |

| Returns  |
| ------------- |
| ModelWrapper  |

Applies a list of QONNX transformations to a given model.
        
If 'validate' is enabled, the function compares the model's output pre and post-transformation. 'transform_list' can accept either a list of transforms, or a nested list of transforms. This affects how model validation is performed. Regular transform-lists are validated after every transform. Nested transform-lists are validated after every sub-list, so transforms [[1,2,3],[4,5]] would be validated between 3-4, and after 5.

---

### `apply_builder_step(model, step, validate, outpu_dir, cfg_settings)`

| parameters  |  Type  |
| ------------- | ------------- |
| `model`  |  `ModelWrapper`  |
| `step`  |  `callable`  |
| `output_dir`  |  `str`  |
| `cfg_settings`  |  `dict` (default = {})  |

| Returns  |
| ------------- |
| ModelWrapper  |

Apply a FINN Builder step to a QONNX ModelWrapper. Takes in the Model, the step function to be executed, and any named parameters of that need to be used in the step's. These named parameters are passed via a dictionary, with the name of each parameter as its key.