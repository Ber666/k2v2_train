import json
import os
import sys
import torch
import types

from utils import get_mcore_transformer_block_key, print_memory_usage
from transformers import LlamaConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


def load_checkpoint(args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    print('=' * 20)
    print(args.load_dir)
    print(args.ckpt_step)
    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir,
                '--position-embedding-type', args.position_embedding_type,
                ]
    if args.ckpt_step > 0:
        sys.argv.extend(['--ckpt-step', str(args.ckpt_step)])

    margs = parse_args()
    margs, checkpoint_args = load_args_from_checkpoint(margs, exit_on_missing_checkpoint=True)

    # Explicitly copy data types from checkpoint.
    margs.fp16 = checkpoint_args.fp16
    margs.bf16 = checkpoint_args.bf16

    # Copy moe args from checkpoint.
    margs.num_experts = checkpoint_args.num_experts
    margs.moe_grouped_gemm = checkpoint_args.moe_grouped_gemm
    margs.qk_layernorm = checkpoint_args.qk_layernorm
    margs.expert_model_parallel_size = checkpoint_args.expert_model_parallel_size
    margs.sequence_parallel = checkpoint_args.sequence_parallel
    margs.moe_aux_loss_coeff = checkpoint_args.moe_aux_loss_coeff
    margs.rotary_base = checkpoint_args.rotary_base


    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    # Validate margs.
    margs = validate_args(margs)

    margs.use_mcore_models = True
    margs.transformer_impl = args.loader_transformer_impl

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_moe_llm360 import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None
    def get_models(tp_size, dtype):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        model_array_len = margs.virtual_pipeline_model_parallel_size
        if model_array_len is None:
            model_array_len = 1
        models = [[] for _ in range(model_array_len)]
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        for rank in range(tp_size):
            mpu.set_tensor_model_parallel_rank(rank)
            if margs.virtual_pipeline_model_parallel_size is not None:
                model_ = []
                for i in range(margs.virtual_pipeline_model_parallel_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(
                        pre_process=pre_process,
                        post_process=post_process
                    ).to(dtype)
                    model_.append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                model_rank = 0
                model_ = [model_provider(pre_process, post_process).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            margs.exit_on_missing_checkpoint = True
            if args.ckpt_step and args.ckpt_step > 0:
                load_checkpoint(model_, None, None, ckpt_step=args.ckpt_step)
            else:
                load_checkpoint(model_, None, None)

            if consumed_train_samples is not None:
                # assert(margs.consumed_train_samples == consumed_train_samples)
                pass
            else:
                consumed_train_samples = margs.consumed_train_samples
            if consumed_valid_samples is not None:
                # assert(margs.consumed_valid_samples == consumed_valid_samples)
                pass
            else:
                consumed_valid_samples = margs.consumed_valid_samples
            for vp_rank in range(model_array_len):
                models[vp_rank].append(model_[vp_rank])  # [[t0, t1, ...]]

            # Print memory usage.
            print_memory_usage("loader", rank, tp_size)

        return models

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    fused_kernels.load(margs)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.use_mcore_models = margs.use_mcore_models
    md.moe_grouped_gemm = margs.moe_grouped_gemm
    md.num_experts = margs.num_experts
    md.group_query_attention = margs.group_query_attention
    md.num_query_groups = margs.num_query_groups
    md.kv_channels = margs.kv_channels
    md.moe_aux_loss_coeff = margs.moe_aux_loss_coeff
    md.rotary_base = margs.rotary_base

    # Get transformer block (named either 'encoder' or 'decoder').
    transformer_block_key = get_mcore_transformer_block_key(md.model_type)
    def get_transformer_block(_model):
        return getattr(_model, transformer_block_key)

    def set_parallel_rng():
        from megatron.core.tensor_parallel.random import initialize_rng_tracker
        initialize_rng_tracker()
        from megatron.core.tensor_parallel.random import _CUDA_RNG_STATE_TRACKER, _DATA_PARALLEL_RNG_TRACKER_NAME, _MODEL_PARALLEL_RNG_TRACKER_NAME, _EXPERT_PARALLEL_RNG_TRACKER_NAME
        _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 123)
        _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 111)
        _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, 333)

    set_parallel_rng()
    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(tp_size, md.params_dtype)]
    models = all_models[0][0]

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples

    hf_state_dict = {}

    # convert embedding
    hf_state_dict["model.embed_tokens.weight"] = torch.cat(
        [models[tp_rank].embedding.word_embeddings.weight.data for tp_rank in range(0, tp_size)],
        dim = 0).clone()
    print("Converted embedding")

    print(models[0])
    print(get_transformer_block(models[0]).layers[0].mlp)


    # convert transformer layers
    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(tp_size, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            num_layer_per_pp = len(get_transformer_block(models[0]).layers)
            for layer_num in range(len(get_transformer_block(models[0]).layers)):
                # Get non-parallel tensors from tp_rank 0
                layer = get_transformer_block(models[0]).layers[layer_num]
                hf_layer_pre = f"model.layers.{layer_num + pp_rank * num_layer_per_pp}"
                hf_state_dict[f"{hf_layer_pre}.input_layernorm.weight"] = layer.self_attention.linear_qkv.layer_norm_weight.data.clone() + 1.0
                hf_state_dict[f"{hf_layer_pre}.post_attention_layernorm.weight"] = layer.mlp.linear_fc1.layer_norm_weight.data.clone() + 1.0

                # Grab all parallel tensors for this layer
                qkv_weight = []
                dense_weight = []
                mlp_gate_weight = []  # element's shape is (ffn_hidden_size / tp, hidden_size)
                mlp_up_weight = []  # element's shape is (ffn_hidden_size / tp, hidden_size)
                mlp_down_weight = []  # element's shape is (hidden_size, ffn_hidden_size / tp)
                for rank in range(0, tp_size):
                    model = models[rank]
                    layer = get_transformer_block(model).layers[layer_num]
                    # Grab attention weights (only have tensor parallel)
                    qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
                    dense_weight.append(layer.self_attention.linear_proj.weight.data)
                    # Grab mlp weights (only have tensor parallel)
                    gate, up = layer.mlp.linear_fc1.weight.data.split(margs.ffn_hidden_size // tp_size, dim=0)
                    mlp_gate_weight.append(gate)
                    mlp_up_weight.append(up)
                    mlp_down_weight.append(layer.mlp.linear_fc2.weight.data)

                print('mlp_gate_weight', len(mlp_gate_weight), mlp_gate_weight[0].shape)

                # Handle mlp units
                hf_gate_weight = torch.cat(mlp_gate_weight, dim=0)
                hf_up_weight = torch.cat(mlp_up_weight, dim=0)
                hf_down_weight = torch.cat(mlp_down_weight, dim=1)

                hf_state_dict[f"{hf_layer_pre}.mlp.gate_proj.weight"] = hf_gate_weight
                hf_state_dict[f"{hf_layer_pre}.mlp.up_proj.weight"] = hf_up_weight
                hf_state_dict[f"{hf_layer_pre}.mlp.down_proj.weight"] = hf_down_weight

                # Handle self-attention params
                num_heads = md.num_attention_heads // tp_size
                num_query_groups = (md.num_query_groups if md.group_query_attention else md.num_attention_heads) // tp_size
                num_querys_per_group = num_heads // num_query_groups
                dim = md.kv_channels
                assert num_heads % num_querys_per_group == 0

                qkv_weight = [i.reshape((num_query_groups, (num_querys_per_group + 2) * dim, md.hidden_size)) for i in qkv_weight]
                hf_q = torch.cat([i[:, :num_querys_per_group * dim, :].reshape(-1, md.hidden_size)
                                for i in qkv_weight])
                hf_kv = [i[:, num_querys_per_group * dim:, :] for i in qkv_weight]
                hf_kv = [torch.chunk(i, 2, dim=1) for i in hf_kv]
                hf_k = torch.cat([i[0].reshape(-1, md.hidden_size) for i in hf_kv])
                hf_v = torch.cat([i[1].reshape(-1, md.hidden_size) for i in hf_kv])

                hf_state_dict[f"{hf_layer_pre}.self_attn.q_proj.weight"] = hf_q
                hf_state_dict[f"{hf_layer_pre}.self_attn.k_proj.weight"] = hf_k
                hf_state_dict[f"{hf_layer_pre}.self_attn.v_proj.weight"] = hf_v
                hf_state_dict[f"{hf_layer_pre}.self_attn.o_proj.weight"] = torch.cat(dense_weight, dim=1)

                print(f"Converted layer {total_layer_num}")
                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    hf_state_dict["model.norm.weight"] = get_transformer_block(models[0]).final_layernorm.weight.data.clone() + 1.0
    hf_state_dict["lm_head.weight"] = torch.cat([models[i].output_layer.weight.data.clone() for i in range(0, tp_size*ep_size, ep_size)])

    print('Finished conversion!!')

    # save config
    def save_config():
        config = LlamaConfig()
        config.architectures = ["LlamaForCausalLM"]
        config.bos_token_id = 0
        config.eos_token_id = 1
        config.hidden_act = "silu"
        config.hidden_size = margs.hidden_size
        config.intermediate_size = margs.ffn_hidden_size
        config.max_position_embeddings = margs.max_position_embeddings
        config.model_type = "llama"
        config.num_attention_heads = margs.num_attention_heads
        config.num_hidden_layers = margs.num_layers
        config.num_key_value_heads = margs.num_query_groups
        config.num_local_experts = margs.num_experts
        config.tie_word_embeddings = not margs.untie_embeddings_and_output_weights
        config.rope_theta = margs.rotary_base
        config.vocab_size = hf_state_dict["model.embed_tokens.weight"].shape[0]
        config.rms_norm_eps = margs.norm_epsilon

        config.save_pretrained(args.save_dir)

    if not os.path.exists(args.save_dir):
        os.system(f'mkdir -p {args.save_dir}')
    save_config()
    print(f"Saved config to {args.save_dir}")

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(hf_state_dict, max_shard_size=max_shard_size)
    # Save model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_dir, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_dir, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_dir, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Megatron Checkpoint Converter Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--ckpt-step', type=int, default=-1, help='checkpoint step')
    parser.add_argument("--max_shard_size", type=str, default="25GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    add_arguments(parser)
    args = parser.parse_args()

    print(args)

    load_checkpoint(args)



if __name__ == '__main__':
    main()

