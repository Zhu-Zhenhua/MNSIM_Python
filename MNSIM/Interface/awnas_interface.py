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
import collections
import torch
from torch import nn
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
        # set self cache
        self.cache = collections.OrderedDict()

    def _get_method_adc(self):
        return "FIX_TRAIN", "SCALE"

    def _get_objective_mode(self, inputs):
        # set net device to inputs device
        self.net.to(inputs.device)
        # get objective mod
        # assert self.objective.mode in ["train_mnsim", "eval"], \
        #     "objective.mode can only be train_mnsim or eval in MNSIM, but {}".format(
        #         self.objective.mode
        #     )
        self.net.eval()
        for _, module in self.net.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # module.train()
                pass

    def origin_evaluate(self, inputs):
        """
        origin evaluate
        """
        method, adc_action = self._get_method_adc()
        self._get_objective_mode(inputs)
        with torch.no_grad():
            outputs = self.net(inputs, method, adc_action)
        return outputs

    def hardware_evaluate(self, inputs):
        """
        hardware evaluate
        """
        _, adc_action = self._get_method_adc()
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
        self.latency_evaluate(disable_inner_pipeline)
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
        if not f"latency_{disable_inner_pipeline}" in self.cache.keys():
            self._get_latency_model()
            if not (disable_inner_pipeline):
                self.latency_model.calculate_model_latency(mode=1)
            else:
                self.latency_model.calculate_model_latency_nopipe()
            self.cache[f"latency_{disable_inner_pipeline}"] = \
                max(max(self.latency_model.finish_time)) / 1e6
        return self.cache[f"latency_{disable_inner_pipeline}"]

    def area_evaluate(self):
        if not "area" in self.cache.keys():
            self._get_area_model()
            self.cache["area"] = self.area_model.arch_total_area / 1e6
            # print(self.area_model.arch_total_xbar_utilization)
        return self.cache["area"]

    def energy_evaluate(self, disable_inner_pipeline=False):
        if not "energy" in self.cache.keys():
            self._get_energy_model(disable_inner_pipeline=disable_inner_pipeline)
            self.cache["energy"] = self.energy_model.arch_total_energy / 1e6
        return self.cache["energy"]

    def power_evaluate(self):
        if not "power" in self.cache.keys():
            self._get_power_model()
            self.cache["power"] = self.power_model.arch_total_power
        return self.cache["power"]
