import math
import configparser
import collections


def load_sim_config(SimConfig_path):
    xbar_config = configparser.ConfigParser()
    xbar_config.read(SimConfig_path, encoding="UTF-8")
    hardware_config = collections.OrderedDict()
    # xbar_size
    xbar_size = list(
        map(int, xbar_config.get("Crossbar level", "Xbar_Size").split(","))
    )
    xbar_row = xbar_size[0]
    xbar_column = xbar_size[1]
    hardware_config["xbar_size"] = xbar_size[0]
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
    input_bit = DAC_precision_dict[DAC_choice]
    quantize_bit = ADC_precision_dict[ADC_choice]
    hardware_config["input_bit"] = input_bit
    hardware_config["quantize_bit"] = quantize_bit
    return hardware_config
