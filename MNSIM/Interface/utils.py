# -*-coding:utf-8-*-
# pylint: disable=R0914
# pylint: disable=C0103
"""
@FileName:
    utils.py
@Description:
    utils for interface
@Authors:
    Chenyu Wang and
    Hanbo Sun
@CreateTime:
    2021/08/03 16:17
"""

import collections
import configparser
import math
from copy import deepcopy
from collections import OrderedDict
import torch

# add quantize_input_flag in input_params
# add quantize_flag in StraightLayer
# add bypass_quantize_weight in conv and fc layers

def load_sim_config(SimConfig_path, extra_define):
    """
    load SimConfig
    return hardware_config, xbar_column, tile_row, tile_column
    """
    xbar_config = configparser.ConfigParser()
    xbar_config.read(SimConfig_path, encoding="UTF-8")
    hardware_config = collections.OrderedDict()
    # xbar_size
    xbar_size = list(
        map(int, xbar_config.get("Crossbar level", "Xbar_Size").split(","))
    )
    xbar_row = xbar_size[0]
    xbar_column = xbar_size[1]
    hardware_config["xbar_size"] = xbar_row
    # xbar bit
    xbar_bit = int(xbar_config.get("Device level", "Device_Level"))
    hardware_config["weight_bit"] = math.floor(math.log2(xbar_bit))
    # input bit and ADC bit
    ADC_choice = int(xbar_config.get("Interface level", "ADC_Choice"))
    DAC_choice = int(xbar_config.get("Interface level", "DAC_Choice"))
    temp_DAC_bit = int(xbar_config.get("Interface level", "DAC_Precision"))
    temp_ADC_bit = int(xbar_config.get("Interface level", "ADC_Precision"))
    ADC_precision_dict = {
        -1: temp_ADC_bit,
        1: 10,
        # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage
        # Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
        2: 8,
        # reference: ISAAC: A Convolutional Neural Network Accelerator with
        # In-Situ Analog Arithmetic in Crossbars
        3: 8,
        # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC
        # with Recursive-Weight DAC
        4: 6,
        # reference: Area-Efficient 1GS/s 6b SAR ADC
        # with Charge-Injection-Cell-Based DAC
        5: 8,  # ASPDAC1
        6: 6,  # ASPDAC2
        7: 4,  # ASPDAC3
    }
    DAC_precision_dict = {
        -1: temp_DAC_bit,
        1: 1,  # 1-bit
        2: 2,  # 2-bit
        3: 3,  # 3-bit
        4: 4,  # 4-bit
        5: 6,  # 6-bit
        6: 8,  # 8-bit
    }
    input_bit = DAC_precision_dict[DAC_choice]
    quantize_bit = ADC_precision_dict[ADC_choice]
    hardware_config["input_bit"] = input_bit
    hardware_config["quantize_bit"] = quantize_bit
    # group num
    # pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
    tile_size = list(map(int, xbar_config.get("Tile level", "PE_Num").split(",")))
    tile_row = tile_size[0]
    tile_column = tile_size[1]
    # extra define
    if extra_define is not None:
        hardware_config["input_bit"] = extra_define["dac_res"]
        hardware_config["quantize_bit"] = extra_define["adc_res"]
        hardware_config["xbar_size"] = extra_define["xbar_size"]
    return hardware_config, xbar_column, tile_row, tile_column


def transfer_awnas_state_dict(cand_net):
    """
    transfer awnas state dict to MNSIM.NetGraph
    inputs: cand_net
    output: MNSIM.NetGraph.state_dict
    """
    mnsim_cfg = cand_net.get_mnsim_cfg(preserve_scale_flag=True)
    param_list = cand_net.weights_manager._param_list
    # stage1: extract param_list from cand_net, add the missing to param_list
    assert len(param_list) == len(mnsim_cfg), \
        "param_list and mnsim_cfg should have the same length"
    for param, cfg in zip(param_list, mnsim_cfg):
        assert len(cfg["output"]) == 1, \
            "ONLY support one output layer"
        if len(cfg["output"][0]) == 2:
            last_value = 1.
        else:
            assert len(cfg["output"][0]) == 3, \
                "output can be in length 2 or 3"
            if cfg["output"][0][1] == float('nan') or \
                cfg["output"][0][1] is float('nan') or \
                cfg["output"][0][1] is None:
                last_value = 1.
            else:
                last_value = cfg["output"][0][1]
        param.update({
            "last_value": torch.Tensor([last_value])
        })
        if cfg["_type"] == "conv" or cfg["_type"] == "fc" or cfg["_type"] == "group_conv":
            assert len(cfg['input']) == 1 and len(cfg['output']) == 1
            assert "weight_info" in cfg.keys()
            bit_scale_list = []
            bit_scale_list.append([
                cfg["input"][0][0],
                cfg["input"][0][1] / (2 ** (cfg["input"][0][0] - 1) - 1),
            ])
            bit_scale_list.append([
                cfg["weight_info"]["bit"],
                cfg["weight_info"]["scale"] / (2 ** (cfg["weight_info"]["bit"] - 1) - 1),
            ])
            bit_scale_list.append([
                cfg["output"][0][0],
                cfg["output"][0][1] / (2 ** (cfg["output"][0][0] - 1) - 1),
            ])
            param.update({
                "bit_scale_list": torch.Tensor(bit_scale_list)
            })
            # save padding flag before
    # state2: transfer keys of param_list into mnsim acceptable keys
    state_dict = collections.OrderedDict()
    assert len(param_list) == len(mnsim_cfg)
    for i, (param, cfg) in enumerate(zip(param_list, mnsim_cfg)):
        state_dict[f"layer_list.{i}.last_value"] = param["last_value"]
        if cfg["_type"] == "fc" or cfg["_type"] == "conv":
            state_dict[f"layer_list.{i}.bit_scale_list"] = param["bit_scale_list"]
            state_dict[f"layer_list.{i}.layer_list.0.weight"] = param["weight"]
            if not cfg["_type"] == "fc":
                state_dict[f"layer_list.{i}.layer_list.0.padding_flag"] = param["padding_flag"]
        # if self.layer_group_conv
        if cfg["_type"] == "group_conv":
            state_dict[f"layer_list.{i}.bit_scale_list"] = param["bit_scale_list"]
            g = cfg["groups"]
            l = param["weight"].shape[0]
            assert l % g == 0
            step = l // g
            for j in range(cfg["groups"]):
                state_dict[f"layer_list.{i}.group_conv.{j}.last_value"] = \
                    param["last_value"]
                state_dict[f"layer_list.{i}.group_conv.{j}.bit_scale_list"] = \
                    param["bit_scale_list"]
                state_dict[f"layer_list.{i}.group_conv.{j}.layer_list.0.weight"] = \
                    param["weight"][(j*step):((j+1)*step),...]
                state_dict[f"layer_list.{i}.group_conv.{j}.layer_list.0.padding_flag"] = \
                    param["padding_flag"]
        if mnsim_cfg[i]["_type"] == "bn":
            state_dict[f"layer_list.{i}.layer.weight"] = param["weight"]
            state_dict[f"layer_list.{i}.layer.bias"] = param["bias"]
            state_dict[f"layer_list.{i}.layer.running_mean"] = param["running_mean"]
            state_dict[f"layer_list.{i}.layer.running_var"] = param["running_var"]
        else:
            for k, _ in param.items():
                assert not k.endswith("bias")
        # check for bias
    return state_dict

def transfer_awnas_layer_list(mnsim_cfg):
    """
    transfer awnas layer list to MNSIM.NetGraph layer_config_list
    inputs: mnsim_cfg
    outputs: MNSIM.NetGraph inputs:
        contain: layer_config_list, quantize_config_list, input_index_list,
            and input_params
    """
    # transfer layer_config_list
    layer_config_list = transfer_layer_config_list(mnsim_cfg)
    # transfer quantize_config_list
    quantize_config_list = list()
    for _, cfg in enumerate(mnsim_cfg):
        assert len(cfg["output"]) == 1, \
            "ONLY support one output layer"
        quantize_config_list.append({
            "weight_bit": cfg["weight_info"]["bit"] \
                if "weight_info" in cfg.keys() \
                else None,
            "activation_bit": cfg["output"][0][0],
            "point_shift": -2,
        })
    # transfer input_params
    assert mnsim_cfg[0]["input"][0] is not None and \
        mnsim_cfg[0]["input"][0][0] is not None and \
        mnsim_cfg[0]["input"][0][1] is not float('nan'), \
        "There should be one quantize layer be first in super net"
    # input shape, 1 be the batch
    input_params = {
        "activation_scale": mnsim_cfg[0]["input"][0][1] \
            / (2 ** (mnsim_cfg[0]["input"][0][0] - 1) - 1),
        "activation_bit": mnsim_cfg[0]["input"][0][0],
        "input_shape": [1] + mnsim_cfg[0]["input"][0][2],
    }
    # add for quantize input
    input_params.update({
        "quantize_input_flag": True
    })
    # transfer input_index_list
    input_index_list = list()
    for _, layer_config in enumerate(layer_config_list):
        if "input_index" in layer_config:
            input_index_list.append(layer_config["input_index"])
        else:
            input_index_list.append([-1])
    return (
        layer_config_list,
        quantize_config_list,
        input_index_list,
        input_params,
    )

def transfer_layer_config_list(mnsim_cfg):
    """
    transfer of layer_config_list is too detailed, so an extra transfer function is defined
    input: mnsim_cfg
    output: layer_config_list
    """
    layer_config_list = list()
    for _, cfg in enumerate(mnsim_cfg):
        layer_config_list.append(OrderedDict())
        if cfg["_type"] == "conv":
            layer_config_list[-1]["type"] = "conv"
            layer_config_list[-1]["in_channels"] = cfg["in_channels"]
            layer_config_list[-1]["out_channels"] = cfg["out_channels"]
            layer_config_list[-1]["kernel_size"] = cfg["kernel_size"]
            layer_config_list[-1]["stride"] = cfg["stride"]
            layer_config_list[-1]["padding"] = cfg["padding"]
        elif cfg["_type"] == "bn":
            layer_config_list[-1]["type"] = "bn"
            layer_config_list[-1]["features"] = cfg["features"]
        elif cfg["_type"] == "fc":
            layer_config_list[-1]["type"] = "fc"
            layer_config_list[-1]["out_features"] = cfg["out_features"]
            layer_config_list[-1]["in_features"] = cfg["in_features"]
        elif cfg["_type"] == "adaptive_pooling":
            layer_config_list[-1]["type"] = "pooling"
            assert cfg["input_size"][0] % cfg["output_size"][0] == 0
            assert cfg["input_size"][1] % cfg["output_size"][1] == 0
            assert cfg["input_size"][0] // cfg["output_size"][0] == \
                cfg["input_size"][1] // cfg["output_size"][1], \
                "kernel_size should be the same along height and width"
            layer_config_list[-1]["mode"] = "AVE"
            layer_config_list[-1]["kernel_size"] = \
                cfg["input_size"][0] // cfg["output_size"][0]
            layer_config_list[-1]["stride"] = layer_config_list[-1]["kernel_size"]
        elif cfg["_type"] == "relu":
            layer_config_list[-1]["type"] = "relu"
        elif cfg["_type"] == "AdaptiveAvgPool2d":
            layer_config_list[-1]["type"] = "pooling"
            assert cfg["input_size"][0] % cfg["output_size"][0] == 0
            assert cfg["input_size"][1] % cfg["output_size"][1] == 0
            assert cfg["input_size"][0] // cfg["output_size"][0] == \
                cfg["input_size"][1] // cfg["output_size"][1], \
                "kernel_size should be the same along height and width"
            layer_config_list[-1]["mode"] = cfg["mode"]
            layer_config_list[-1]["kernel_size"] = \
                cfg["input_size"][0] // cfg["output_size"][0]
            layer_config_list[-1]["stride"] = layer_config_list[-1]["kernel_size"]
        elif cfg["_type"] == "flatten":
            layer_config_list[-1]["type"] = "flatten"
            layer_config_list[-1]["start_dim"] = cfg["start_dim"]
            layer_config_list[-1]["end_dim"] = cfg["end_dim"]
        elif cfg["_type"] == "hard_tanh":
            layer_config_list[-1]["type"] = "hard_tanh"
        elif cfg["_type"] == "element_sum":
            layer_config_list[-1]["type"] = "element_sum"
        elif cfg["_type"] == "expand":
            layer_config_list[-1]["type"] = "expand"
            layer_config_list[-1]["max_channels"] =cfg["max_channels"]
        elif cfg["_type"] == "downsample":
            layer_config_list[-1]["type"] = "downsample"
        elif cfg["_type"] == "group_conv":
            layer_config_list[-1]["type"] = "group_conv"
            layer_config_list[-1]["in_channels"] = cfg["in_channels"]
            layer_config_list[-1]["out_channels"] = cfg["out_channels"]
            layer_config_list[-1]["kernel_size"] = cfg["kernel_size"]
            layer_config_list[-1]["stride"] = cfg["stride"]
            layer_config_list[-1]["padding"] = cfg["padding"]
            layer_config_list[-1]["groups"] = cfg["groups"]
        elif cfg["_type"] == "quantize":
            raise Exception(
                "NOT support quantize layer in MNSIM"
            )
        else:
            raise Exception(
                "NOT support type {}".format(cfg["_type"])
            )
        #transfer input_index
        layer_config_list[-1]["input_index"] = list()
        for _, from_pos in enumerate(cfg["from"]):
            layer_config_list[-1]["input_index"].append(
                from_pos - cfg["to"][0]
            )
        # append for StraightLayer quantize
        # check the output scale is all the same input scale
        assert len(cfg["output"]) == 1
        assert len(cfg["output"][0]) == 3
        for input_info in cfg["input"]:
            assert len(input_info) == 3, \
                "output should be 1, and all input, output should have 3 item"
        output_scale = cfg["output"][0][1]
        quantize_flag = False
        for input_info in cfg["input"]:
            input_scale = input_info[1]
            if input_scale == output_scale \
                or input_scale is output_scale:
                continue
            else:
                quantize_flag = True
        if cfg["_type"] in ["conv", "fc", "group_conv"]:
            # append for QuantizeLayer to pass quantizing weights
            # since weights are already quantized
            layer_config_list[-1]["bypass_quantize_weight"] = True
        else:
            # StraightLayer for quantize output
            layer_config_list[-1]["quantize_flag"] = quantize_flag
    return layer_config_list
