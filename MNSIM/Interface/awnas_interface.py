# -*-coding:utf-8-*-
import collections
import configparser
import copy
import math
import copy
import torch

import torch.nn as nn

import numpy as np

from importlib import import_module

from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.interface import mysplit


class awnas_TrainTestInterface(object):
    def __init__(
        self,
        dataset_module_name,
        SimConfig_path,
        net,
        objective_mode,
        _param_list=None,
        device=None,
        extra_define=None,
        test_loader=None,
    ):

        self.dataset_module_name = dataset_module_name
        self.test_loader = test_loader
        self.set_sim_config(SimConfig_path)
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]
        self.objective_mode = objective_mode
        # net and weights
        self.set_device(device)
        if extra_define != None:
            self.hardware_config["input_bit"] = extra_define["dac_res"]
            self.hardware_config["quantize_bit"] = extra_define["adc_res"]
            self.hardware_config["xbar_size"] = extra_define["xbar_size"]
        self.net = net
        if _param_list is not None:
            state_dict = self.net.get_state_dict(_param_list)
            self.net.load_change_weights(state_dict)

    def set_sim_config(self, SimConfig_path):
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding="UTF-8")
        self.hardware_config = collections.OrderedDict()
        # xbar_size
        xbar_size = list(
            map(int, xbar_config.get("Crossbar level", "Xbar_Size").split(","))
        )
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config["xbar_size"] = xbar_size[0]
        # xbar bit
        self.xbar_bit = int(xbar_config.get("Device level", "Device_Level"))
        self.hardware_config["weight_bit"] = math.floor(math.log2(self.xbar_bit))
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get("Interface level", "ADC_Choice"))
        DAC_choice = int(xbar_config.get("Interface level", "DAC_Choice"))
        temp_DAC_bit = int(xbar_config.get("Interface level", "DAC_Precision"))
        temp_ADC_bit = int(xbar_config.get("Interface level", "ADC_Precision"))
        ADC_precision_dict = {
            -1: temp_ADC_bit,
            1: 10,
            # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
            2: 8,
            # reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
            3: 8,  # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
            4: 6,  # reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
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
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.quantize_bit = ADC_precision_dict[ADC_choice]
        self.hardware_config["input_bit"] = self.input_bit
        self.hardware_config["quantize_bit"] = self.quantize_bit
        # group num
        self.pe_group_num = int(xbar_config.get("Process element level", "Group_Num"))
        self.tile_size = list(
            map(int, xbar_config.get("Tile level", "PE_Num").split(","))
        )
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]

    def set_device(self, device):
        if device != None:
            self.device = torch.device(
                f"{device}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

    def set_mode(self):
        self.net.to(self.device)
        if self.objective_mode.startswith("train"):
            self.net.train()
        elif self.objective_mode.startswith("eval"):
            self.net.eval()
        else:
            assert "unsupported objective_mode"

    def origin_evaluate(self, method="SINGLE_FIX_TEST", adc_action="SCALE"):
        self.set_mode()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if i > 10:
                    break
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net(images, method, adc_action)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total

    def get_net_bits(self):
        net_bit_weights = self.net.get_weights()
        return net_bit_weights

    def set_net_bits_evaluate(self, net_bit_weights, adc_action="SCALE"):
        self.set_mode()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if i > 10:
                    break
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net.set_weights_forward(
                    images, net_bit_weights, adc_action
                )
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total

    def get_structure(self):
        net_bit_weights = self.net.get_weights()
        net_structure_info = self.net.get_structure()
        assert len(net_bit_weights) == len(net_structure_info)
        # set relative index to absolute index
        absolute_index = [None] * len(net_structure_info)
        absolute_count = 0
        for i in range(len(net_structure_info)):
            if not (
                len(net_structure_info[i]["Outputindex"]) == 1
                and net_structure_info[i]["Outputindex"][0] == 1
            ):
                raise Exception("duplicate output")
            if net_structure_info[i]["type"] in [
                "conv",
                "pooling",
                "element_sum",
                "fc",
            ]:
                absolute_index[i] = absolute_count
                absolute_count = absolute_count + 1
            else:
                if not len(net_structure_info[i]["Inputindex"]) == 1:
                    raise Exception("duplicate input index")
                absolute_index[i] = absolute_index[
                    i + net_structure_info[i]["Inputindex"][0]
                ]
        graph = list()
        for i in range(len(net_structure_info)):
            if net_structure_info[i]["type"] in [
                "conv",
                "pooling",
                "element_sum",
                "fc",
            ]:
                # layer num, layer type
                layer_num = absolute_index[i]
                layer_type = net_structure_info[i]["type"]
                # layer input
                layer_input = list(
                    map(
                        lambda x: (absolute_index[i + x] if i + x != -1 else -1),
                        net_structure_info[i]["Inputindex"],
                    )
                )
                # layer output
                layer_output = list()
                for tmp_i in range(len(net_structure_info)):
                    if net_structure_info[tmp_i]["type"] in [
                        "conv",
                        "pooling",
                        "element_sum",
                        "fc",
                    ]:
                        tmp_layer_num = absolute_index[tmp_i]
                        tmp_layer_input = list(
                            map(
                                lambda x: (
                                    absolute_index[tmp_i + x] if tmp_i + x != -1 else -1
                                ),
                                net_structure_info[tmp_i]["Inputindex"],
                            )
                        )
                        if layer_num in tmp_layer_input:
                            layer_output.append(tmp_layer_num)
                graph.append((layer_num, layer_type, layer_input, layer_output))
        # add to net array
        net_array = []
        for layer_num, (layer_bit_weights, layer_structure_info) in enumerate(
            zip(net_bit_weights, net_structure_info)
        ):
            # change layer structure info
            layer_structure_info = copy.deepcopy(layer_structure_info)
            layer_count = absolute_index[layer_num]
            layer_structure_info["Layerindex"] = graph[layer_count][0]
            layer_structure_info["Inputindex"] = list(
                map(lambda x: x - graph[layer_count][0], graph[layer_count][2])
            )
            layer_structure_info["Outputindex"] = list(
                map(lambda x: x - graph[layer_count][0], graph[layer_count][3])
            )
            # add for element_sum and pooling
            if layer_bit_weights == None:
                if layer_structure_info["type"] in ["element_sum", "pooling"]:
                    net_array.append([(layer_structure_info, None)])
                continue
            assert (
                len(layer_bit_weights.keys())
                == layer_structure_info["row_split_num"]
                * layer_structure_info["weight_cycle"]
                * 2
            )
            # split
            for i in range(layer_structure_info["row_split_num"]):
                for j in range(layer_structure_info["weight_cycle"]):
                    layer_bit_weights[f"split{i}_weight{j}_positive"] = mysplit(
                        layer_bit_weights[f"split{i}_weight{j}_positive"],
                        self.xbar_column,
                    )
                    layer_bit_weights[f"split{i}_weight{j}_negative"] = mysplit(
                        layer_bit_weights[f"split{i}_weight{j}_negative"],
                        self.xbar_column,
                    )
            # generate pe array
            xbar_array = []
            for i in range(layer_structure_info["row_split_num"]):
                L = len(layer_bit_weights[f"split{i}_weight{0}_positive"])
                for j in range(L):
                    pe_array = []
                    for s in range(layer_structure_info["weight_cycle"]):
                        pe_array.append(
                            [
                                layer_bit_weights[f"split{i}_weight{s}_positive"][
                                    j
                                ].astype(np.uint8),
                                layer_bit_weights[f"split{i}_weight{s}_negative"][
                                    j
                                ].astype(np.uint8),
                            ]
                        )
                    xbar_array.append(pe_array)
            # store in xbar_array
            total_array = []
            L = math.ceil(len(xbar_array) / (self.tile_row * self.tile_column))
            for i in range(L):
                tile_array = []
                for h in range(self.tile_row):
                    for w in range(self.tile_column):
                        serial_number = (
                            i * self.tile_row * self.tile_column
                            + h * self.tile_column
                            + w
                        )
                        if serial_number < len(xbar_array):
                            tile_array.append(xbar_array[serial_number])
                total_array.append((layer_structure_info, tile_array))
            net_array.append(total_array)
        return net_array


class awnas_NetworkGraph(NetworkGraph):
    def __init__(
        self,
        hardware_config,
        layer_config_list,
        quantize_config_list,
        input_index_list,
        input_params,
    ):
        super(awnas_NetworkGraph, self).__init__(
            hardware_config,
            layer_config_list,
            quantize_config_list,
            input_index_list,
            input_params,
        )

    def get_state_dict(self, _param_list):
        state_dict = collections.OrderedDict()
        for i in range(len(_param_list)):
            if (
                self.layer_list[i].layer_config["type"] == "fc"
                or self.layer_list[i].layer_config["type"] == "conv"
            ):
                state_dict[f"layer_list.{i}.bit_scale_list"] = _param_list[i][
                    "bit_scale_list"
                ]
                if "bias" in _param_list[i].keys():
                    _param_list[i].pop("bias")
                if "weight" in _param_list[i].keys():
                    state_dict[f"layer_list.{i}.layer_list.{i}.weight"] = _param_list[
                        i
                    ]["weight"]
            # if self.layer_con
            if self.layer_list[i].layer_config["type"] == "bn":
                state_dict[f"layer_list.{i}.layer.weight"] = _param_list[i]["weight"]
                state_dict[f"layer_list.{i}.layer.bias"] = _param_list[i]["bias"]
                state_dict[f"layer_list.{i}.layer.running_mean"] = _param_list[i][
                    "running_mean"
                ]
                state_dict[f"layer_list.{i}.layer.running_var"] = _param_list[i][
                    "running_var"
                ]
            state_dict[f"layer_list.{i}.last_value"] = _param_list[i]["last_value"]
        return state_dict
