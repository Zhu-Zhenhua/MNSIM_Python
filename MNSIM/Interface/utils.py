#-*-coding:utf-8-*-
# pylint: disable=R0914
# pylint: disable=C0103
'''
@FileName:
    utils.py
@Description:
    utils for interface
@Authors:
    Chenyu Wang and
    Hanbo Sun
@CreateTime:
    2021/08/03 16:17
'''

import collections
import configparser
import math


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
    tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
    tile_row = tile_size[0]
    tile_column = tile_size[1]
    # extra define
    if extra_define is not None:
        hardware_config['input_bit'] = extra_define['dac_res']
        hardware_config['quantize_bit'] = extra_define['adc_res']
        hardware_config['xbar_size'] = extra_define['xbar_size']
    return hardware_config, xbar_column, tile_row, tile_column


def transfer_awnas_state_dict(layer_list, param_list):
    """
    transfer awnas state dict to MNSIM.NetGraph
    inputs: layer_list and param_list
    output: MNSIM.NetGraph.state_dict
    """
    state_dict = collections.OrderedDict()
    for i in range(len(param_list)):
        if (
            layer_list[i].layer_config["type"] == "fc"
            or layer_list[i].layer_config["type"] == "conv"
        ):
            state_dict[f"layer_list.{i}.bit_scale_list"] = param_list[i][
                "bit_scale_list"
            ]
            if "bias" in param_list[i].keys():
                param_list[i].pop("bias")
            if "weight" in param_list[i].keys():
                state_dict[f"layer_list.{i}.layer_list.{i}.weight"] = param_list[
                    i
                ]["weight"]
        # if self.layer_con
        if layer_list[i].layer_config["type"] == "bn":
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

def transfer_awnas_layer_list(layer_list):
    """
    transfer awnas layer list to MNSIM.NetGraph layer_config_list
    inputs: awnas layer list
    outputs: MNSIM.NetGraph inputs:
        contain: hardware_config, layer_config_list, quantize_config_list, input_index_list,
            and input_params
    """
    hardware_config = None
    layer_config_list = None
    quantize_config_list = None
    input_index_list = None
    input_params = None
    return hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params
