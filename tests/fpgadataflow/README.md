<a id="module-op_test"></a>

<a id="op-test-module"></a>

# op_test module

<a id="op_test.OpTest"></a>

### *class* op_test.OpTest

Bases: `ABC`

An abstract class which uses PyTest’s functionality to make writing tests for Brainsmith operators easier.

> This class contains tests and fixtures, which are features of PyTest.
> - All methods beginning with `f_` are fixtures, i.e. [`OpTest.f_model()`](#op_test.OpTest.f_model). See [PyTest’s fixture documentation](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html) for more info.
> - All methods beginning with `test_` are tests, i.e. [`OpTest.test_cycles()`](#op_test.OpTest.test_cycles). See [PyTest’s getting started page](https://docs.pytest.org/en/stable/) for more info.
> - All methods that do not have these prefixes are helper functions, to be used to aid in writing your own tests.

<a id="op_test.OpTest.f_model"></a>

#### *abstract* f_model() → ModelWrapper

An abstract fixture that generates the QONNX ModelWrapper to be tested (when
implemented). Each test MUST override this fixture, otherwise any PyTests
will result in a NotImplementedError. Helper functions such as create_model()
and run_transforms() may be useful in reducing boilerplate when implementing this
fixture.

* **Returns:**
  A `ModelWrapper` containing the ONNX graph we’ll use for testing
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`

<a id="op_test.OpTest.f_model_hw"></a>

#### f_model_hw(f_model: ModelWrapper, f_infer_hw_transform: Transformation, f_save_models: bool, f_output_dir: str) → ModelWrapper

Converts all ONNX layers of a specific type to hardware layers,
using a given inference function. If that function does not exist
(i.e. f_infer_hw_transform == ‘None’), then the model is directly
passed through. All fixtures reliant on hardware inference should
check if f_infer_hw_transform == ‘None’ before using this fixture.

* **Parameters:**
  * **f_model** (`qonnx.core.modelwrapper.ModelWrapper`) – Auto-populated by the [`OpTest.f_model()`](#op_test.OpTest.f_model) fixture’s return value
  * **f_infer_hw_transform** – Auto-populated by the [`OpTest.f_infer_hw_transform()`](#op_test.OpTest.f_infer_hw_transform) fixture’s return value
* **Returns:**
  A `ModelWrapper` containing the converted model
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`

<a id="op_test.OpTest.f_model_specialised"></a>

#### f_model_specialised(f_model_hw: ModelWrapper, f_target_fpga: str, f_output_dir: str) → ModelWrapper

A fixture that applies layer specialisation to the ‘f_model_hw’
fixture, then returns the resulting ModelWrapper.;

* **Parameters:**
  * **f_model_hw** (`qonnx.core.modelwrapper.ModelWrapper`) – Auto-populated by the [`OpTest.f_model_hw()`](#op_test.OpTest.f_model_hw) fixture’s return value
  * **f_target_fpga** (*str*) – Auto-populated by the [`OpTest.f_target_fpga()`](#op_test.OpTest.f_target_fpga) fixture’s return value
  * **f_output_dir** (*str*) – Auto-populated by the [`OpTest.f_output_dir()`](#op_test.OpTest.f_output_dir) fixture’s return value
* **Returns:**
  A `ModelWrapper` containing the specialised model
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`

<a id="op_test.OpTest.f_infer_hw_transform"></a>

#### f_infer_hw_transform() → Transformation

The transformation to infer a hardware layer from a standard ONNX layer.
If this fixture returns ‘None’, OpTest assumes to skip hardware inference.

* **Returns:**
  The `Transformation` we’ll apply when inferring hardware layers
  (Default: `None`)
* **Return type:**
  `qonnx.transformation.base.Transformation`

<a id="op_test.OpTest.f_target_fpga"></a>

#### f_target_fpga() → str

The fpga we’re targeting for testing. Can be overridden by test classes.

* **Returns:**
  The name of the fpga we’re targeting for testing
  (Default: `"xcv80-lsva4737-2MHP-e-S"`)
* **Return type:**
  str

<a id="op_test.OpTest.f_target_node"></a>

#### f_target_node() → int

The index of the node in the model we’re focusing on. Allows for multiple
nodes to be present, with tests that only target a specific node. Defaults to
the first node. Can be overridden.

* **Returns:**
  The index of the node we wish to focus our tests on.
  (Default: `0`)
* **Return type:**
  int

<a id="op_test.OpTest.f_input_tensors"></a>

#### f_input_tensors(f_model: ModelWrapper) → dict[str, any]

Creates the tensor(s) passed to the model, to be used by the simulation during
testing. By default, this fixture creates a tensor with random values, but can be
overridden by tests to pass specific values.

* **Parameters:**
  **f_model** (`qonnx.core.modelwrapper.ModelWrapper`) – Auto-populated by the [`OpTest.f_model()`](#op_test.OpTest.f_model) fixture’s return value
* **Returns:**
  A dictionary. Each entry in the dictionary contains an input tensor’s name as its key, and the data we wish to pass to it as its value.
* **Return type:**
  `qonnx.transformation.base.Transformation`

<a id="op_test.OpTest.f_save_models"></a>

#### f_save_models() → bool

If this fixture is overridden to return True, the ‘f_auto_saver’ fixture
will automatically save the outputs of all ‘f_model’ fixtures to .onnx files.

* **Returns:**
  Whether or not the auto-saver should save intermediate models.
  (Default: `False`)
* **Return type:**
  bool

<a id="op_test.OpTest.f_auto_saver"></a>

#### f_auto_saver(request: FixtureRequest, f_save_models: bool, f_output_dir: str) → None

Saves the output of each intermediate model step to a .onnx file.

If the ‘f_save_models’ fixture evaluates to true, this fixture
automatically scrapes all fixtures with names that start with ‘f_model’
(i.e. ‘f_model’, ‘f_model_specialised’) and saves their output to a .onnx file
(if the fixtures return a ModelWrapper).

* **Parameters:**
  * **request** (`pytest.fixtures.FixtureRequest`) – Auto-populated by the `pytest.fixtures.FixtureRequest()` fixture’s return value
  * **f_save_models** (*bool*) – Auto-populated by the [`OpTest.f_save_models()`](#op_test.OpTest.f_save_models) fixture’s return value
  * **f_output_dir** (*str*) – Auto-populated by the [`OpTest.f_output_dir()`](#op_test.OpTest.f_output_dir) fixture’s return value

<a id="op_test.OpTest.f_output_dir"></a>

#### f_output_dir() → str

The directory we’ll save the output of our tests to. By default,
OpTest saves to a directory called “test_output”, which is created
in the same directory as the python file containing the test.

* **Returns:**
  The output directory that all test output will be saved to.
  (Default: `"__file__/test_output"`)
* **Return type:**
  str

<a id="op_test.OpTest.f_create_output_dir"></a>

#### f_create_output_dir(f_output_dir: str) → None

Automatically makes a test output directory if none exists. This
fixture auto-runs before all test functions. If the directory exists,
this fixture does nothing.

* **Parameters:**
  **f_output_dir** (*str*) – Auto-populated by the [`OpTest.f_output_dir()`](#op_test.OpTest.f_output_dir) fixture’s return value

<a id="op_test.OpTest.f_exec_mode"></a>

#### f_exec_mode(request) → str

Parameterised fixture. Provides the exec mode parameter (“cppsim or rtlsim”)

* **Returns:**
  The exec mode
  (Default: `"cppsim"` or `"rtlsim"`)
* **Return type:**
  str

<a id="op_test.OpTest.test_cycles"></a>

#### test_cycles(f_model_specialised: ModelWrapper, f_target_node: int, f_exec_mode: str) → None

Ensure the number of cycles the layer takes to run in rtlsim aligns
with the expected number of cycles.

* **Parameters:**
  * **f_model_specialised** (`qonnx.core.modelwrapper.ModelWrapper`) – Auto-populated by the [`OpTest.f_model_specialised()`](#op_test.OpTest.f_model_specialised) fixture’s return value
  * **f_target_node** (*int*) – Auto-populated by the [`OpTest.f_target_node()`](#op_test.OpTest.f_target_node) fixture’s return value
  * **f_exec_mode** (*str*) – Auto-populated by OpTest’s f_exec_mode PyTest parameter.
    These are defined at the top of OpTest’s class definition.

<a id="op_test.OpTest.create_model"></a>

#### create_model(inputs: List[tuple[dict[str, any], str]], outputs: List[tuple[dict[str, any], str]], inits: List[dict[str, any]], nodes: List[dict[str, any]], opset: int = 17, name: str = 'OpTest_Graph') → ModelWrapper

Creates a model using ONNX.helper and QONNX.ModelWrapper. Removes the need for
additional boilerplate code.

* **Parameters:**
  * **inputs** (*List* *(**tuple* *(**dict* *,* *str* *)* *)*) – A list of tuples, with each tuple representing an input of the
    model. These tuples have two elements: the first element in each tuple is a dictionary
    containing the named parameters you’re passing to `onnx.helper.make_tensor_value_info()`.
    The second element is the `QONNX.core.datatype` string that that input will be set to.
  * **outputs** (*List* *(**tuple* *(**dict* *,* *str* *)* *)*) – A list of tuples, with each tuple representing an output of the
    model. These tuples have two elements: the first element in each tuple is a dictionary
    containing  any named parameters you’re passing to `onnx.helper.make_tensor_value_info()`.
    The second element is the `QONNX.core.datatype` string that that input will be set to.
  * **inits** (*List* *(**dict* *)*) – A list of dictionaries, each dictionary representing an initialiser of the model. These should named parameters of ``onnx.numpy_helper.from_array` <[https://onnx.ai/onnx/api/numpy_helper.html#onnx.numpy_helper.to_array](https://onnx.ai/onnx/api/numpy_helper.html#onnx.numpy_helper.to_array)>\`_.
  * **nodes** (*List* *(**dict* *)*) – A list of dictionaries, each dictionary representing a node of the model. These should named parameters of ``onnx.numpy_helper.from_array` <[https://onnx.ai/onnx/api/helper.html#onnx.helper.make_node](https://onnx.ai/onnx/api/helper.html#onnx.helper.make_node)>\`_.
  * **opset** (*int* *,* *optional*) – The opset of the generated model. Defaults to 17.
    (Default: `17`)
  * **name** (*str* *,* *optional*) – The name of the generated model.
    (Default: `"OpTest_Graph"`)
* **Returns:**
  A `ModelWrapper` containing the generated model
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`

<a id="op_test.OpTest.apply_transforms"></a>

#### apply_transforms(model: ModelWrapper, transform_list: List[Transformation] | List[List[Transformation]], input_tensors: dict | None = None, tolerance: float = 1e-05, output_dir: str | None = None) → ModelWrapper

Applies a list of QONNX transformations to a given model.

If ‘validate’ is enabled, the function compares the model’s output pre and
post-transformation. ‘transform_list’ can accept either a list of transforms,
or a nested list of transforms. This affects how model validation is performed.
Regular transform-lists are validated after every transform. Nested transform-
lists are validated after every sub-list, so transforms [[1,2,3],[4,5]] would
be validated between 3-4, and after 5.

If an ‘output_dir’ is provided, the model will be saved to that directory after
ANY transform is applied.

#### WARNING
As of 11/06/2025, validation has stopped working. I’m unsure what commit caused
this. When the ONNX runtime is used to execute the model, the exception “Rounding
error is too high to match set QONNX datatype (INT8) for input xxxxx”, where
“xxxxxx” is a six digit alphanumeric string.

* **Parameters:**
  * **model** (`qonnx.core.modelwrapper.ModelWrapper`) – The `ModelWrapper` we’ll apply our transform list to
  * **transform_list** (*List* *(**Transformation* *) or* *List* *(**List* *(**Transformation* *)* *)*) – A list (or nested list) containing each `Transformation`
    we’ll apply to the model
  * **input_tensors** (*dict*) – The dictionary containing the names of tensors (key)
    and their inputs as numpy arrays (value) which we’d use to validate our model.
    Providing an input dictionary enables model validation.
    (Default: `None`)
  * **tolerance** (*float*) – The acceptable tolerance that model outputs can differ during validation.
    (Default: `1e-5`)
  * **output_dir** (*float*) – The acceptable tolerance that model outputs can differ during validation.
    (Default: `1e-5`)
* **Returns:**
  A `ModelWrapper` containing the transformed model
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`

<a id="op_test.OpTest.apply_builder_step"></a>

#### apply_builder_step(model: ModelWrapper, step: callable, output_dir: str, cfg_settings: dict = {}) → ModelWrapper

Apply a FINN Builder step to a QONNX ModelWrapper. Takes in the Model,
the step function to be executed, and any named parameters of that need to be
used in the step’s DataflowBuildConfig. These named parameters are passed via a
dictionary, with the name of each parameter as its key.

* **Parameters:**
  * **model** (`qonnx.core.modelwrapper.ModelWrapper`) – The `ModelWrapper` we’ll apply our builder step to
  * **step** (*callable*) – A list (or nested list) containing each `Transformation` we’ll
    apply to the model
  * **output_dir** (*str*) – The directory that the builder step will use, if the step
    involves saving intermediate models
  * **cfg_settings** (*dict* *,* *optional*) – A dictionary containing named parameters to pass to the step’s
    `finn.builder.build_dataflow_config.DataFlowBuildConfig`
* **Returns:**
  A `ModelWrapper` containing the transformed model
* **Return type:**
  `qonnx.core.modelwrapper.ModelWrapper`
