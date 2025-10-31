import torch
import torchrec

from torchrec.models.dlrm import DLRM, DLRM_DCN
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if ext_dist.my_size > 1:
                if i not in self.local_emb_indices:
                    continue
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True, padding_idx=m-1)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                if n_emb < ext_dist.my_size:
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, ext_dist.my_size)
                    )
                self.n_global_emb = n_emb
                self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
                    n_emb
                )
                self.local_emb_slice = ext_dist.get_my_slice(n_emb)
                self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        # print(ly)
        return ly



def main():

    m_spa = 2
    ln_emb = torch.tensor([4, 3, 2])
    ln_bot = torch.tensor([4, 3, 2])
    ln_top = torch.tensor([8, 4, 1, 2])
    arch_interaction_op = "dot"
    arch_interaction_itself = False
    sync_dense_params = True
    loss_threshold = 0.0
    ndevices = -1
    qr_flag = False
    qr_operation = "mult"
    qr_collisions = 0
    qr_threshold = 200
    md_flag = False
    md_threshold = 200
    weighted_pooling = None
    loss_function = "mse"

    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=arch_interaction_op,
        arch_interaction_itself=arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=sync_dense_params,
        loss_threshold=loss_threshold,
        ndevices=ndevices,
        qr_flag=qr_flag,
        qr_operation=qr_operation,
        qr_collisions=qr_collisions,
        qr_threshold=qr_threshold,
        md_flag=md_flag,
        md_threshold=md_threshold,
        weighted_pooling=weighted_pooling,
        loss_function=loss_function,
    )

    """
    # workaround 1: tensor -> list
    if torch.is_tensor(lS_i_onnx):
        lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
    # workaound 2: list -> tensor
    lS_i_onnx = torch.stack(lS_i_onnx)
    """
    # debug prints
    print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
    print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))
    dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
    batch_size = X_onnx.shape[0]
    print("X_onnx.shape", X_onnx.shape)
    if torch.is_tensor(lS_o_onnx):
        print("lS_o_onnx.shape", lS_o_onnx.shape)
    else:
        for oo in lS_o_onnx:
            print("oo.shape", oo.shape)
    if torch.is_tensor(lS_i_onnx):
        print("lS_i_onnx.shape", lS_i_onnx.shape)
    else:
        for ii in lS_i_onnx:
            print("ii.shape", ii.shape)

    # name inputs and outputs
    o_inputs = (
        ["offsets"]
        if torch.is_tensor(lS_o_onnx)
        else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
    )
    i_inputs = (
        ["indices"]
        if torch.is_tensor(lS_i_onnx)
        else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
    )
    all_inputs = ["dense_x"] + o_inputs + i_inputs
    # debug prints
    print("inputs", all_inputs)

    # create dynamic_axis dictionaries
    do_inputs = (
        [{"offsets": {1: "batch_size"}}]
        if torch.is_tensor(lS_o_onnx)
        else [
            {"offsets_" + str(i): {0: "batch_size"}} for i in range(len(lS_o_onnx))
        ]
    )
    di_inputs = (
        [{"indices": {1: "batch_size"}}]
        if torch.is_tensor(lS_i_onnx)
        else [
            {"indices_" + str(i): {0: "batch_size"}} for i in range(len(lS_i_onnx))
        ]
    )
    dynamic_axes = {"dense_x": {0: "batch_size"}, "pred": {0: "batch_size"}}
    for do in do_inputs:
        dynamic_axes.update(do)
    for di in di_inputs:
        dynamic_axes.update(di)
    # debug prints
    print(dynamic_axes)
    # export model
    torch.onnx.export(
        dlrm,
        (X_onnx, lS_o_onnx, lS_i_onnx),
        dlrm_pytorch_onnx_file,
        verbose=True,
        opset_version=11,
        input_names=all_inputs,
        output_names=["pred"],
        dynamic_axes=dynamic_axes,
        dynamo=True,
    )
    # Define model parameter

    # embedding_dim = 128
    # num_embeddings_per_feature = [2 for _ in range(26)]
    # eb_configs = [
    #         EmbeddingBagConfig(
    #             name=f"t_{feature_name}",
    #             embedding_dim=embedding_dim,
    #             num_embeddings=num_embeddings_per_feature[feature_idx],
    #             feature_names=[feature_name],
    #         )
    #         for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    # ]
    # # Initialize the DLRM model
    # dlrm_model = DLRM_DCN(
    #      embedding_bag_collection=EmbeddingBagCollection(
    #             tables=eb_configs, device=torch.device("cpu")
    #         ),
    #         dense_in_features=len(DEFAULT_INT_NAMES),
    #         dense_arch_layer_sizes=[512, 256, 128],
    #         over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
    #         dcn_num_layers=3,
    #         dcn_low_rank_dim=512,
    #         dense_device=torch.device("cpu"),
    # )


    # class DLRM_ONNX_WRAPPER(DLRM):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)

    #     def forward(self, dense_features, keys, values, offsets):

    #         sparse_features = torchrec.KeyedJaggedTensor.from_offsets_sync(
    #             keys=keys,
    #             values=values,
    #             offsets=offsets,
    #         )
    #         return super().forward(dense_features, sparse_features).squeeze(-1)

    # B = 2
    # D = 8

    # eb1_config = EmbeddingBagConfig(
    #     name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1"]
    # )
    # eb2_config = EmbeddingBagConfig(
    #     name="t2",
    #     embedding_dim=D,
    #     num_embeddings=100,
    #     feature_names=["f2"],
    # )

    # ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
    # model = DLRM_ONNX_WRAPPER(
    #     embedding_bag_collection=ebc,
    #     dense_in_features=100,
    #     dense_arch_layer_sizes=[20, D],
    #     over_arch_layer_sizes=[5, 1],
    # )

    # features = torch.rand((B, 100))

    # #     0       1
    # # 0   [1,2] [4,5]
    # # 1   [4,3] [2,9]
    # # ^
    # # feature
    # from torchrec import KeyedJaggedTensor
    # sparse_features = KeyedJaggedTensor.from_offsets_sync(
    #     keys=["f1", "f2"],
    #     values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
    #     offsets=torch.tensor([0, 2, 4, 6, 8]),
    # )

    # import pdb; pdb.set_trace()
    # logits = model(
    #     dense_features=features,
    #     keys=["f1", "f2"],
    #     values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
    #     offsets=torch.tensor([0, 2, 4, 6, 8]),
    # )

    # print(logits)
    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         (features, ["f1", "f2"], torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]), torch.tensor([0, 2, 4, 6, 8])),
    #         "dlrm.onnx",
    #         input_names=["dense_features", "keys", "values", "offsets"],
    #         output_names=["logits"],
    #         opset_version=18,
    #         dynamo=True,
    #         report=True
    #         #optimize=True,
    #     )

    # import onnx

    # proto = onnx.load("dlrm.onnx")
    # onnx.checker.check_model(proto)
    # onnx.shape_inference.infer_shapes(proto)
    # onnx.save(proto, "dlrm2.onnx")


    # Wrap the DLRM model with DLRM_DCN for recommendation tasks
    #dlrm_rec_model = DLRM_DCN(dlrm_model)

    # Create dummy input data
    #atch_size = 4
    #dense_input = torch.randn(batch_size, num_dense_features)
    #sparse_input = torch.randint(0, 1000, (batch_size, num_sparse_features))

    # Forward pass through the model
    #output = dlrm_model(dense_input, sparse_input)

    #print("Model output:", output)

if __name__ == "__main__":
    main()
