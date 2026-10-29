# Source port from lululxvi/deepxde; LGPL-2.1.
# Pinned commit and original source/notice: third_party/ch03_upstream/deepxde/.
# Class bodies below extracted verbatim. Minimal single-backend compatibility
# helpers replace backend configuration and activation/initializer registries.
import torch
from abc import ABC, abstractmethod
from types import SimpleNamespace

def _activation(name):
    if callable(name): return name
    return {"tanh": torch.tanh, "relu": torch.relu, "gelu": torch.nn.functional.gelu}[name]
def _initializer(name):
    return {"Glorot normal": torch.nn.init.xavier_normal_,
            "Glorot uniform": torch.nn.init.xavier_uniform_, "zeros": torch.nn.init.zeros_}[name]
activations = SimpleNamespace(get=_activation)
initializers = SimpleNamespace(get=_initializer)
config = SimpleNamespace(real=lambda module: torch.get_default_dtype())

class NN(torch.nn.Module):
    """Base class for all neural network modules."""

    def __init__(self):
        super().__init__()
        self.regularizer = None
        self._auxiliary_vars = None
        self._input_transform = None
        self._output_transform = None

    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)
class FNN(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)

        if isinstance(dropout_rate, list):
            if len(layer_sizes) - 1 != len(dropout_rate):
                raise ValueError(
                    f"Number of dropout rates must be equal to {len(layer_sizes) - 1}"
                )
            self.dropout_rate = dropout_rate
        else:
            self.dropout_rate = [dropout_rate] * (len(layer_sizes) - 1)

        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
            if self.dropout_rate[j] > 0:
                x = torch.nn.functional.dropout(
                    x, p=self.dropout_rate[j], training=self.training
                )
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
class DeepONetStrategy(ABC):
    """DeepONet building strategy.

    See the section 3.1.6. in
    L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. Karniadakis.
    A comprehensive and fair comparison of two neural operators
    (with practical extensions) based on FAIR data.
    Computer Methods in Applied Mechanics and Engineering, 393, 114778, 2022.
    """

    def __init__(self, net):
        self.net = net

    @abstractmethod
    def build(self, layer_sizes_branch, layer_sizes_trunk):
        """Build branch and trunk nets."""

    @abstractmethod
    def call(self, x_func, x_loc):
        """Forward pass."""

class SingleOutputStrategy(DeepONetStrategy):
    """Single output build strategy is the standard build method."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        branch = self.net.build_branch_net(layer_sizes_branch)
        trunk = self.net.build_trunk_net(layer_sizes_trunk)
        return branch, trunk

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = self.net.merge_branch_trunk(x_func, x_loc, 0)
        return x

class IndependentStrategy(DeepONetStrategy):
    """Directly use n independent DeepONets,
    and each DeepONet outputs only one function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        single_output_strategy = SingleOutputStrategy(self.net)
        branch, trunk = [], []
        for _ in range(self.net.num_outputs):
            branch_, trunk_ = single_output_strategy.build(
                layer_sizes_branch, layer_sizes_trunk
            )
            branch.append(branch_)
            trunk.append(trunk_)
        return branch, trunk

    def call(self, x_func, x_loc):
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = self.net.branch[i](x_func)
            x_loc_ = self.net.activation_trunk(self.net.trunk[i](x_loc))
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
        return self.net.concatenate_outputs(xs)

class SplitBothStrategy(DeepONetStrategy):
    """Split the outputs of both the branch net and the trunk net into n groups,
    and then the kth group outputs the kth solution.

    For example, if n = 2 and both the branch and trunk nets have 100 output neurons,
    then the dot product between the first 50 neurons of
    the branch and trunk nets generates the first function,
    and the remaining 50 neurons generate the second function.
    """

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        single_output_strategy = SingleOutputStrategy(self.net)
        return single_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func and x_loc into respective outputs
        shift = 0
        size = x_func.shape[1] // self.net.num_outputs
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)

class SplitBranchStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_branch[-1] / self.net.num_outputs != layer_sizes_trunk[-1]:
            raise AssertionError(
                f"Output size of the trunk net does not equal to {layer_sizes_branch[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_func into respective outputs
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)

class SplitTrunkStrategy(DeepONetStrategy):
    """Split the trunk net and share the branch net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_trunk[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the trunk net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_trunk[-1] / self.net.num_outputs != layer_sizes_branch[-1]:
            raise AssertionError(
                f"Output size of the branch net does not equal to {layer_sizes_trunk[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc):
        x_func = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        # Split x_loc into respective outputs
        shift = 0
        size = x_func.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            x_loc_ = x_loc[:, shift : shift + size]
            x = self.net.merge_branch_trunk(x_func, x_loc_, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)
class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
        dropout_rate: If `dropout_rate` is a ``float`` between 0 and 1, then the
            same rate is used in both trunk and branch nets. If `dropout_rate`
            is a ``dict``, then the trunk net uses the rate `dropout_rate["trunk"]`,
            and the branch net uses `dropout_rate["branch"]`. Both `dropout_rate["trunk"]`
            and `dropout_rate["branch"]` should be ``float`` or lists of ``float``.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularization

        if isinstance(dropout_rate, dict):
            self.dropout_rate_branch = dropout_rate["branch"]
            self.dropout_rate_trunk = dropout_rate["trunk"]
        else:
            self.dropout_rate_branch = self.dropout_rate_trunk = dropout_rate

        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError(
                    "num_outputs is set to 1, but multi_output_strategy is not None."
                )
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                'Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(
            layer_sizes_branch,
            self.activation_branch,
            self.kernel_initializer,
            dropout_rate=self.dropout_rate_branch,
        )

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(
            layer_sizes_trunk,
            self.activation_trunk,
            self.kernel_initializer,
            dropout_rate=self.dropout_rate_trunk,
        )

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
