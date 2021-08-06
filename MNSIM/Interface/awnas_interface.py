# -*-coding:utf-8-*-
"""
@FileName:
    awnas_interface.py
@Description:
    interface between awnas and mnsim
@Authors:
    Chenyu Wang and
    Hanbo Sun
@CreateTime:
    2021/08/03 17:05
"""
# -*-coding:utf-8-*-
import torch
from MNSIM.Interface import utils
from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.interface import TrainTestInterface


class AWNASTrainTestInterface(TrainTestInterface):
    """
    awnas interface
    """

    def __init__(
        self, objective, cand_net, SimConfig_path, extra_define=None, **kwargs
    ):
        # link objective
        self.objective = objective
        # link cand_net
        self.cand_net = cand_net
        # load simulation config
        (
            self.hardware_config,
            self.xbar_column,
            self.tile_row,
            self.tile_column,
        ) = utils.load_sim_config(SimConfig_path, extra_define)
        # input awnas layer list and param list
        (
            layer_config_list,
            quantize_config_list,
            input_index_list,
            input_params,
        ) = utils.transfer_awnas_layer_list(cand_net.get_mnsim_cfg())
        # TODO: use hardware_config searched from awnas to evaluate
        self.net = NetworkGraph(
            self.hardware_config,
            layer_config_list,
            quantize_config_list,
            input_index_list,
            input_params,
        )
        # load weights
        weights = utils.transfer_awnas_state_dict(cand_net)
        self.net.load_change_weights(weights)

    def _get_mothod_adc(self):
        return "FIX_TRAIN", "SCALE"

    def _get_objective_mode(self, inputs):
        # set net device to inputs device
        self.net.to(inputs.device)
        # get objective mod
        if self.objective.mode == "eval":
            self.net.eval()
        else:
            self.net.train()

    def origin_evaluate(self, inputs):
        """
        origin evaluate
        """
        method, adc_action = self._get_mothod_adc()
        self._get_objective_mode(inputs)
        with torch.no_grad():
            outputs = self.net(inputs, method, adc_action)
        return outputs

    def hardware_evaluate(self, inputs):
        """
        hardware evaluate
        """
        _, adc_action = self._get_mothod_adc()
        self._get_objective_mode(inputs)
        with torch.no_grad():
            net_weights = self.get_net_bits()
            # add variation to net weights
            outputs = self.net.set_weights_forward(inputs, net_weights, adc_action)
        return outputs
