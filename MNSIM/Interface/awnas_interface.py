# -*-coding:utf-8-*-
import collections
import configparser
import copy
import math
import os
import copy
import re
import torch

import torch.nn as nn

import numpy as np

from importlib import import_module
from MNSIM.Interface import quantize


class awnas_TrainTestInterface(object):
    def __init__(
        self,
        network_module,
        dataset_module_name,
        SimConfig_path,
        net,
        _param_list=None,
        device=None,
        extra_define=None,
        test_loader=None,
    ):
        # network_module = 'lenet'
        # dataset_module_name = 'cifar10'
        # _param_list = './zoo/cifar10_lenet_train_params.pth'
        # load net, dataset, and weights
        self.network_module = network_module
        self.dataset_module_name = dataset_module_name
        self._param_list = _param_list
        self.test_loader = test_loader
        # load simconfig
        ## xbar_size, input_bit, weight_bit, quantize_bit
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
        # net and weights
        if device != None:
            self.device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        # print(f'run on device {self.device}')
        if dataset_module_name.endswith("cifar10"):
            num_classes = 10
        elif dataset_module_name.endswith("cifar100"):
            num_classes = 100
        else:
            assert 0, f"unknown dataset"
        if extra_define != None:
            self.hardware_config["input_bit"] = extra_define["dac_res"]
            self.hardware_config["quantize_bit"] = extra_define["adc_res"]
            self.hardware_config["xbar_size"] = extra_define["xbar_size"]
        self.net = net
        if _param_list is not None:
            # 保存下权重
            self.net.load_change_weights(_param_list)

    def origin_evaluate(self, method="SINGLE_FIX_TEST", adc_action="SCALE"):
        self.net.to(self.device)
        self.net.eval()
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
        self.net.to(self.device)
        self.net.eval()
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
        # test index
        # graph = map(lambda x: x[0][0],net_array)
        # graph = list(map(lambda x: f'l: {x["Layerindex"]}, t: {x["type"]}, i: {x["Inputindex"]}, o: {x["Outputindex"]}', graph))
        # graph = '\n'.join(graph)
        return net_array


def mysplit(array, length):
    # reshape
    array = np.reshape(array, (array.shape[0], -1))
    # split on output
    assert array.shape[0] > 0
    tmp_index = []
    for i in range(1, array.shape[0]):
        if i % length == 0:
            tmp_index.append(i)
    return np.split(array, tmp_index, axis=0)


class awnas_NetworkGraph(nn.Module):
    def __init__(
        self,
        hardware_config,
        layer_config_list,
        quantize_config_list,
        input_index_list,
        input_params,
    ):
        super(awnas_NetworkGraph, self).__init__()
        # same length for layer_config_list , quantize_config_list and input_index_list
        assert len(layer_config_list) == len(quantize_config_list)
        assert len(layer_config_list) == len(input_index_list)
        # layer list
        self.layer_list = nn.ModuleList()
        # add layer to layer list by layer_config, quantize_config, and input_index
        for layer_config, quantize_config in zip(
            layer_config_list, quantize_config_list
        ):
            assert "type" in layer_config.keys()
            if layer_config["type"] in quantize.QuantizeLayerStr:
                layer = quantize.QuantizeLayer(
                    hardware_config, layer_config, quantize_config
                )
            elif layer_config["type"] in quantize.StraightLayerStr:
                layer = quantize.StraightLayer(
                    hardware_config, layer_config, quantize_config
                )
            else:
                assert 0, f'not support {layer_config["type"]}'
            self.layer_list.append(layer)
        # save input_index_list, input_index is a list
        self.input_index_list = copy.deepcopy(input_index_list)
        self.input_params = copy.deepcopy(input_params)

    def forward(self, x, method="SINGLE_FIX_TEST", adc_action="SCALE"):
        # input fix information
        quantize.last_activation_scale = self.input_params["activation_scale"]
        quantize.last_activation_bit = self.input_params["activation_bit"]
        # forward
        tensor_list = [x]
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            if len(input_index) == 1:
                tensor_list.append(
                    layer.forward(
                        tensor_list[input_index[0] + i + 1], method, adc_action
                    )
                )
            else:
                tensor_list.append(
                    layer.forward(
                        [
                            tensor_list[input_index[0] + i + 1],
                            tensor_list[input_index[1] + i + 1],
                        ],
                        method,
                        adc_action,
                    )
                )
            if i == (len(self.layer_list) - 3):
                print("yaotule")
        return tensor_list[-1]

    def get_weights(self):
        net_bit_weights = []
        for layer in self.layer_list:
            net_bit_weights.append(layer.get_bit_weights())
        return net_bit_weights

    def set_weights_forward(self, x, net_bit_weights, adc_action="SCALE"):
        # input fix information
        quantize.last_activation_scale = self.input_params["activation_scale"]
        quantize.last_activation_bit = self.input_params["activation_bit"]
        # filter None
        net_bit_weights = list(filter(lambda x: x != None, net_bit_weights))
        # forward
        tensor_list = [x]
        count = 0
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            if isinstance(layer, quantize.QuantizeLayer):
                tensor_list.append(
                    layer.set_weights_forward(
                        tensor_list[input_index[0] + i + 1],
                        net_bit_weights[count],
                        adc_action,
                    )
                )
                # tensor_list.append(layer.forward(tensor_list[input_index[0] + i + 1], 'SINGLE_FIX_TEST', adc_action))
                count = count + 1
            else:
                if len(input_index) == 1:
                    tensor_list.append(
                        layer.forward(
                            tensor_list[input_index[0] + i + 1], "FIX_TRAIN", None
                        )
                    )
                else:
                    tensor_list.append(
                        layer.forward(
                            [
                                tensor_list[input_index[0] + i + 1],
                                tensor_list[input_index[1] + i + 1],
                            ],
                            "FIX_TRAIN",
                            None,
                        )
                    )
        return tensor_list[-1]

    def get_structure(self):
        # forward structure
        x = torch.zeros(self.input_params["input_shape"])
        self.to(x.device)
        self.eval()
        tensor_list = [x]
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            if len(input_index) == 1:
                tensor_list.append(
                    layer.structure_forward(tensor_list[input_index[0] + i + 1])
                )
            else:
                tensor_list.append(
                    layer.structure_forward(
                        [
                            tensor_list[input_index[0] + i + 1],
                            tensor_list[input_index[1] + i + 1],
                        ],
                    )
                )
        # structure information, stored as list
        net_info = []
        for layer in self.layer_list:
            net_info.append(layer.layer_info)
        return net_info

    def load_change_weights(self, _param_list):
        # input is a state dict, weights
        # concat all layer_list weights
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

        keys_map = collections.OrderedDict()
        for key in state_dict.keys():
            tmp_key = re.sub("\.layer_list\.\d+\.weight$", "", key)
            if tmp_key not in keys_map.keys():
                keys_map[tmp_key] = [key]
            else:
                keys_map[tmp_key].append(key)
        # concat and split
        tmp_state_dict = collections.OrderedDict()
        for tmp_key, key_list in keys_map.items():
            if len(key_list) == 1 and tmp_key == key_list[0]:
                tmp_state_dict[tmp_key] = state_dict[key_list[0]]
            else:
                # get layer info
                layer_config = None
                hardware_config = None
                for i in range(len(self.layer_list)):
                    name = f"layer_list.{i}"
                    if name == tmp_key:
                        layer_config = self.layer_list[i].layer_config
                        hardware_config = self.layer_list[i].hardware_config
                assert layer_config, "layer must have layer config"
                assert hardware_config, "layer must have hardware config"
                # concat weights
                if layer_config["type"] != "bn":
                    total_weights = torch.cat(
                        [state_dict[key] for key in key_list], dim=1
                    )
                    # total_weights=state_dict[key_list[0]]
                    # split weights

                    if layer_config["type"] == "conv":
                        split_len = hardware_config["xbar_size"] // (
                            layer_config["kernel_size"] ** 2
                        )
                    elif layer_config["type"] == "fc":
                        split_len = hardware_config["xbar_size"]
                    else:
                        assert 0, f'not support {layer_config["type"]}'
                    weights_list = torch.split(total_weights, split_len, dim=1)
                    # load weights
                for i, weights in enumerate(weights_list):
                    tmp_state_dict[tmp_key + f".layer_list.{i}.weight"] = weights
        # load weights
        self.load_state_dict(tmp_state_dict)
