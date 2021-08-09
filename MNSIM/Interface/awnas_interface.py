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
import math
from MNSIM.Interface import utils
from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Model_latency import Model_latency
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy

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
        # store SimConfig for hardware simulation
        self.SimConfig_path = SimConfig_path
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
        # get TCG_mapping for hardware simulation
        self.structure_file = self.get_structure()
        self.TCG_mapping = TCG(self.structure_file, self.SimConfig_path)
        # 4 hardware simulation components
        self.latency_model = None
        self.power_model = None
        self.area_model = None
        self.energy_model = None

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

    def _get_latency_model(self):
        if self.latency_model is not None:
            return
        self.latency_model = Model_latency(
            NetStruct=self.structure_file,
            SimConfig_path=self.SimConfig_path,
            TCG_mapping=self.TCG_mapping
        )

    def _get_area_model(self):
        if self.area_model is not None:
            return
        self.area_model = Model_area(
            NetStruct=self.structure_file,
            SimConfig_path=self.SimConfig_path,
            TCG_mapping=self.TCG_mapping
        )

    def _get_energy_model(self, disable_inner_pipeline=False):
        if self.power_model is not None:
            return
        # self._get_latency_model()
        self.latency_evaluate()
        self._get_power_model()
        self.energy_model = Model_energy(
            NetStruct=self.structure_file,
            SimConfig_path=self.SimConfig_path,
            TCG_mapping=self.TCG_mapping,
            model_latency=self.latency_model,
            model_power=self.power_model
        )

    def _get_power_model(self):
        if self.power_model is not None:
            return
        self.power_model = Model_inference_power(
            NetStruct=self.structure_file,
            SimConfig_path=self.SimConfig_path,
            TCG_mapping=self.TCG_mapping
        )

    def latency_evaluate(self, disable_inner_pipeline=False):
        self._get_latency_model()
        if not (disable_inner_pipeline):
            self.latency_model.calculate_model_latency(mode=1)
        else:
            self.latency_model.calculate_model_latency_nopipe()
        total_latency = sum(self.latency_model.total_buffer_latency) + \
            sum(self.latency_model.total_computing_latency) + \
            sum(self.latency_model.total_digital_latency) + \
            sum(self.latency_model.total_intra_tile_latency) + \
            sum(self.latency_model.total_inter_tile_latency)
        return total_latency

    def area_evaluate(self):
        self._get_area_model()
        return self.area_model.arch_total_area

    def energy_evaluate(self, disable_inner_pipeline=False):
        self._get_energy_model(disable_inner_pipeline=disable_inner_pipeline)
        return self.energy_model.arch_total_energy

    def power_evaluate(self):
        self._get_power_model()
        return self.power_model.arch_total_power
