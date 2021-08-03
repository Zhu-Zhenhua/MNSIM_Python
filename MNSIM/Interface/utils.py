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
    mnsim_cfg = cand_net.get_mnsim_cfg()
    param_list = list()
    param_list = cand_net.weights_manager._param_list
    # stage1: extract param_list from cand_net,add the missing to param_list
    for i in range(len(param_list)):
        param_list[i].update(
            {
                "last_value": torch.FloatTensor(
                    [
                        mnsim_cfg[i]["output"][1]
                        if mnsim_cfg[i]["output"][1] is not None
                        else 1
                    ]
                )
            }
        )
        if mnsim_cfg[i]["_type"] == "conv" or mnsim_cfg[i]["_type"] == "fc":
            if "weight_info" in mnsim_cfg[i].keys():
                param_list[i].update(
                    {
                        "bit_scale_list": torch.FloatTensor(
                            [
                                [
                                    mnsim_cfg[i]["input"][0],
                                    mnsim_cfg[i]["input"][1] / (2 ** (mnsim_cfg[i]["input"][0] - 1) - 1),
                                ],
                                [
                                    mnsim_cfg[i]["weight_info"]["bit"],
                                    mnsim_cfg[i]["weight_info"]["scale"] / (2 ** (mnsim_cfg[i]["weight_info"]["bit"] - 1) - 1),
                                ],
                                [
                                    mnsim_cfg[i]["output"][0],
                                    mnsim_cfg[i]["output"][1] / (2 ** (mnsim_cfg[i]["output"][0] - 1) - 1),
                                ],
                            ]
                        )
                    }
                )
            else:
                param_list[i].update(
                    {
                        "bit_scale_list": torch.FloatTensor(
                            [
                                [
                                    mnsim_cfg[i]["input"][0],
                                    mnsim_cfg[i]["input"][1] / (2 ** (mnsim_cfg[i]["input"][0] - 1) - 1),
                                ],
                                [None, None],
                                [
                                    mnsim_cfg[i]["output"][0],
                                    mnsim_cfg[i]["output"][1] / (2 ** (mnsim_cfg[i]["output"][0] - 1) - 1),
                                ],
                            ]
                        )
                    }
                )
    # state2: transfer keys of param_list into mnsim acceptable keys
    state_dict = collections.OrderedDict()
    for i in range(len(param_list)):
        if mnsim_cfg[i]["_type"] == "fc" or mnsim_cfg[i]["_type"] == "conv":
            state_dict[f"layer_list.{i}.bit_scale_list"] = param_list[i][
                "bit_scale_list"
            ]
            if "bias" in param_list[i].keys():
                param_list[i].pop("bias")
            if "weight" in param_list[i].keys():
                state_dict[f"layer_list.{i}.layer_list.{i}.weight"] = param_list[i][
                    "weight"
                ]
        # if self.layer_con
        if mnsim_cfg[i]["_type"] == "bn":
            state_dict[f"layer_list.{i}.layer.weight"] = param_list[i]["weight"]
            state_dict[f"layer_list.{i}.layer.bias"] = param_list[i]["bias"]
            state_dict[f"layer_list.{i}.layer.running_mean"] = param_list[i][
                "running_mean"
            ]
            state_dict[f"layer_list.{i}.layer.running_var"] = param_list[i][
                "running_var"
            ]
        state_dict[f"layer_list.{i}.last_value"] = param_list[i]["last_value"]
    return state_dict


def transfer_awnas_layer_list(mnsim_cfg):
    """
    transfer awnas layer list to MNSIM.NetGraph layer_config_list
    inputs: mnsim_cfg
    outputs: MNSIM.NetGraph inputs:
        contain: hardware_config, layer_config_list, quantize_config_list, input_index_list,
            and input_params
    """
    # transfer layer_config_list
    layer_config_list = transfer_layer_config_list(mnsim_cfg)
    # transfer quantize_config_list
    quantize_config_list = list()
    for i in range(len(mnsim_cfg)):
        quantize_config_list.append(
            {
                "weight_bit": mnsim_cfg[i]["weight_info"]["bit"]
                if "weight_info" in mnsim_cfg[i].keys()
                else None,
                "activation_bit": mnsim_cfg[i]["output"][0],
                "point_shift": -2,
            }
        )
    # transfer input_params
    input_params = OrderedDict()
    if mnsim_cfg[0]["input"][0] is not None and mnsim_cfg[0]["input"][0] is not None:
        input_params = {
            "activation_scale": mnsim_cfg[0]["input"][1]
            / (2 ** (mnsim_cfg[0]["input"][0] - 1) - 1),
            "activation_bit": mnsim_cfg[0]["input"][0],
            "input_shape": (1, 3, 32, 32),
        }
    # transfer input_index_list
    input_index_list = list()
    for i in range(len(layer_config_list)):
        if "input_index" in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]["input_index"])
        else:
            input_index_list.append([-1])
    # transfer hardware_config TODO
    hardware_config = None
    return (
        hardware_config,
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
    for i in range(len(mnsim_cfg)):
        if mnsim_cfg[i]["_type"] == "conv":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "conv"
            layer_config_list[-1]["in_channels"] = mnsim_cfg[i]["in_channels"]
            layer_config_list[-1]["out_channels"] = mnsim_cfg[i]["out_channels"]
            layer_config_list[-1]["kernel_size"] = mnsim_cfg[i]["kernel_size"]
            layer_config_list[-1]["stride"] = mnsim_cfg[i]["stride"]
            layer_config_list[-1]["padding"] = mnsim_cfg[i]["padding"]
        elif mnsim_cfg[i]["_type"] == "bn":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "bn"
            layer_config_list[-1]["features"] = mnsim_cfg[i]["features"]
        elif mnsim_cfg[i]["_type"] == "fc":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "fc"
            layer_config_list[-1]["out_features"] = mnsim_cfg[i]["out_features"]
            layer_config_list[-1]["in_features"] = mnsim_cfg[i]["in_features"]
        elif mnsim_cfg[i]["_type"] == "relu":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "relu"
        elif mnsim_cfg[i]["_type"] == "element_sum":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "element_sum"
        elif mnsim_cfg[i]["_type"] == "AdaptiveAvgPool2d":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "AdaptiveAvgPool2d"
            layer_config_list[-1]["mode"] = "AVE"
            layer_config_list[-1]["output_size"] = mnsim_cfg[i]["output_size"]
        elif mnsim_cfg[i]["_type"] == "flatten":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "flatten"
            layer_config_list[-1]["start_dim"] = mnsim_cfg[i]["start_dim"]
            layer_config_list[-1]["end_dim"] = mnsim_cfg[i]["end_dim"]
        elif mnsim_cfg[i]["_type"] == "expand":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "expand"
            layer_config_list[-1]["_max_channels"] = mnsim_cfg[i]["_max_channels"]
        elif mnsim_cfg[i]["_type"] == "downsample":
            layer_config_list.append(OrderedDict())
            layer_config_list[-1]["type"] = "downsample"
        else:
            assert 0, "not support type {}".format(mnsim_cfg[i]["_type"])
        #transfer input_index
        layer_config_list[-1]["input_index"] = list()
        for _, from_pos in enumerate(mnsim_cfg[i]["from"]):
            layer_config_list[-1]["input_index"].append(
                from_pos - mnsim_cfg[i]["to"][0]
            )
    return layer_config_list
