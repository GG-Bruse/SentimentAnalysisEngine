import argparse
import numpy as np
import tensorrt as trt
import os

from configer import BertConfig
from load_weight import BertLoader

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
plg_registry = trt.get_plugin_registry()
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "1", "")
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "1", "")

"""
Attentions Keys
"""
WQ = "self_query_kernel"
BQ = "self_query_bias"
WK = "self_key_kernel"
BK = "self_key_bias"
WV = "self_value_kernel"
BV = "self_value_bias"
WQKV = "self_qkv_kernel"
BQKV = "self_qkv_bias"
"""
Transformer Keys
"""
W_AOUT = "attention_output_dense_kernel"
B_AOUT = "attention_output_dense_bias"
AOUT_LN_BETA = "attention_output_layernorm_beta"
AOUT_LN_GAMMA = "attention_output_layernorm_gamma"
W_MID = "intermediate_dense_kernel"
B_MID = "intermediate_dense_bias"
W_LOUT = "output_dense_kernel"
B_LOUT = "output_dense_bias"
LOUT_LN_BETA = "output_layernorm_beta"
LOUT_LN_GAMMA = "output_layernorm_gamma"



def get_mha_dtype(config):
    dtype = trt.float32
    if config.use_fp16:
        dtype = trt.float16
    return int(dtype)



def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name

def set_output_name(layer, prefix, name, out_idx = 0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)



def emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes):
    # 动态维度(batch_size, sequence_length)
    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1, -1))
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1, -1))
    input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(-1, -1))
    for batch_size in sorted(batch_sizes):
        if len(sequence_lengths) == 1:
            profile = builder.create_optimization_profile()
            min_shape = (1, sequence_lengths[0])
            shape = (batch_size, sequence_lengths[0])
            profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
            profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
            profile.set_shape("input_mask", min=min_shape, opt=shape, max=shape)
            builder_config.add_optimization_profile(profile)
        else:
            for sequence_length in sorted(sequence_lengths):
                profile = builder.create_optimization_profile()
                min_shape = (1, sequence_length)
                shape = (batch_size, sequence_length)
                profile.set_shape("input_ids", min=min_shape, opt=shape, max=shape)
                profile.set_shape("segment_ids", min=min_shape, opt=shape, max=shape)
                profile.set_shape("input_mask", min=min_shape, opt=shape, max=shape)
                builder_config.add_optimization_profile(profile)

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"].numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"].numpy(), trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"].numpy(), trt.PluginFieldType.FLOAT32)
    output_fp16 = trt.PluginField("output_fp16", np.array([1 if config.use_fp16 else 0]).astype(np.int32), trt.PluginFieldType.INT32)
    mha_type = trt.PluginField("mha_type_id", np.array([get_mha_dtype(config)], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16, mha_type])
    fn = emln_plg_creator.create_plugin("embeddings", pfc)
    
    input_ids = network.add_shuffle(input_ids)
    input_ids.second_transpose = (1, 0)
    segment_ids = network.add_shuffle(segment_ids)
    segment_ids.second_transpose = (1, 0)
    input_mask = network.add_shuffle(input_mask)
    input_mask.second_transpose = (1, 0)
    inputs = [input_ids.get_output(0),
              segment_ids.get_output(0),
              input_mask.get_output(0)]
    emb_layer = network.add_plugin_v2(inputs, fn)

    set_output_name(emb_layer, "embeddings_", "output")
    return emb_layer



def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    assert(len(input_tensor.shape) == 5)
    B, S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)
    set_output_name(mult_all, prefix, "qkv_mult")

    has_mask = imask is not None
    # QKV2CTX
    pf_type = trt.PluginField("type_id", np.array([get_mha_dtype(config)], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0)]
    if has_mask:
        qkv_in.append(imask)
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)

    set_output_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx



def skipln(prefix, config, init_dict, network, input_tensor, skip, bias=None):
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]
    if config.use_fp16:
        dtype = trt.float16

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "beta"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "gamma"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]

    if bias:
        pf_bias = trt.PluginField("bias", bias.numpy(), trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer



def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    context_transposed = attention_layer_opt(prefix + "attention_", config, init_dict, network, input_tensor, imask)
    attention_heads = context_transposed.get_output(0)
    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    W_aout = init_dict[prefix + W_AOUT]
    attention_out_fc = network.add_fully_connected(attention_heads, hidden_size, W_aout, B_aout)
    skiplayer = skipln(prefix + "attention_output_layernorm_",config, init_dict, network, attention_out_fc.get_output(0), input_tensor, B_aout)
    attention_ln = skiplayer.get_output(0)
     # FC1 + GELU
    B_mid = init_dict[prefix + B_MID]
    W_mid = init_dict[prefix + W_MID]
    mid_dense = network.add_fully_connected(attention_ln, config.intermediate_size, W_mid, B_mid)
    mid_dense_out = mid_dense.get_output(0)
    # GELU
    POW = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(mid_dense_out, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(mid_dense_out, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), mid_dense_out, trt.ElementWiseOperation.PROD)
    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")
    # FC2
    # Dense to hidden size
    B_lout = init_dict[prefix + B_LOUT]
    W_lout = init_dict[prefix + W_LOUT]
    out_dense = network.add_fully_connected(intermediate_act, hidden_size, W_lout, B_lout)
    set_output_name(out_dense, prefix + "output_", "dense")

    out_layer = skipln(prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), attention_ln, B_lout)
    set_output_name(out_layer, prefix + "output_", "reshape")

    return out_layer



def bert_model(config, init_dict, network, input_tensor, input_mask):
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        out_layer = transformer_layer_opt(ss, config,  init_dict, network, prev_input, input_mask)
        prev_input = out_layer.get_output(0)
    return prev_input



def squad_output(prefix, config, init_dict, network, input_tensor):
    idims = input_tensor.shape
    assert len(idims) == 5
    _, _, hidden_size, _, _ = idims

    p_w = init_dict["bert_pooler_dense_kernel"]
    p_b = init_dict["bert_pooler_dense_bias"]
    pool_output = network.add_fully_connected(input_tensor, hidden_size, p_w, p_b)
    pool_data = pool_output.get_output(0)
    tanh = network.add_activation(pool_data, trt.tensorrt.ActivationType.TANH)
    tanh_output = tanh.get_output(0)
    W_out = init_dict["output_weights"]
    B_out = init_dict["output_bias"]

    n_values = int(os.environ["output_numbers"])
    dense = network.add_fully_connected(tanh_output, n_values, W_out, B_out)
    dense_data = dense.get_output(0)
    sigmoid = network.add_activation(dense_data, trt.tensorrt.ActivationType.SIGMOID)

    set_output_name(sigmoid, prefix, 'sigmoid')
    return sigmoid 



def build_engine(batch_sizes, workspace_size, sequence_lengths, config, weights_dict):
    # 显性批处理
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = workspace_size * (1024 * 1024)
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        emb_layer = emb_layernorm(builder, network, config, weights_dict, builder_config, sequence_lengths, batch_sizes)
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)

        bert_out = bert_model(config, weights_dict, network, embeddings, mask_idx)
        squad_logits = squad_output("cls_", config, weights_dict, network, bert_out)
        squad_logits_out = squad_logits.get_output(0)
        network.mark_output(squad_logits_out)
        engine = builder.build_engine(network, builder_config)
        return engine

def main(args):
    config = BertConfig(args.config_file)
    batch_size = [5]
    workspace_size = 10000
    sequence_length = [512]

    loader = BertLoader(config)
    loader.load_weight(args.ckpt_file)
    loader.load_weight_bias_dict()
    weights_dict = loader.weights_dict

    with build_engine(batch_size, workspace_size, sequence_length, config, weights_dict) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output_path))
        with open(args.output_path, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use_fp16', default=True)
    argparser.add_argument('--config_file', default='../model/config.json')
    argparser.add_argument('--output_path', default='../output/model.engine')
    argparser.add_argument('-ckpt', '--ckpt_file',  required=True)
    args = argparser.parse_args()
    main(args)
