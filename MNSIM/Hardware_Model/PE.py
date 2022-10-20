#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
import sys
work_path = os.path.join(os.path.dirname(os.getcwd()),"MNSIM_Python-Master/")
# print(work_path)
sys.path.append(work_path)
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Hardware_Model.DAC import DAC
from MNSIM.Hardware_Model.ADC import ADC
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.ShiftReg import shiftreg
from MNSIM.Hardware_Model.Reg import reg
from MNSIM.Hardware_Model.Buffer import buffer
test_SimConfig_path = os.path.join(work_path,"SimConfig.ini")
# print(test_SimConfig_path)
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class ProcessElement(crossbar, DAC, ADC):
	def __init__(self, SimConfig_path):
		crossbar.__init__(self, SimConfig_path)
		DAC.__init__(self, SimConfig_path)
		ADC.__init__(self, SimConfig_path)
		PE_config = cp.ConfigParser()
		PE_config.read(SimConfig_path, encoding='UTF-8')
		self.PIM_type_pe = int(PE_config.get('Process element level', 'PIM_Type'))
		self.sub_position = 0
		__xbar_polarity = int(PE_config.get('Process element level', 'Xbar_Polarity'))
		# self.PE_multiplex_xbar_num = list(
		# 	map(int, PE_config.get('Process element level', 'Multiplex_Xbar_Num').split(',')))
		if __xbar_polarity == 1:
			self.PE_multiplex_xbar_num = [1,1]
		else:
			assert __xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
			self.PE_multiplex_xbar_num = [1,2]
			self.sub_position = int(PE_config.get('Process element level', 'Sub_Position'))
		self.group_num = int(PE_config.get('Process element level', 'Group_Num'))
		if self.group_num == 0:
			self.group_num = 1
		self.num_occupied_group = 0
		self.PE_xbar_num = self.group_num * self.PE_multiplex_xbar_num[0] * self.PE_multiplex_xbar_num[1]
		# self.polarity = PE_config.get('Algorithm Configuration', 'Weight_Polarity')
		# if self.polarity == 2:
		# 	assert self.PE_xbar_num[1]%2 == 0
		self.PE_simulation_level = int(PE_config.get('Algorithm Configuration', 'Simulation_Level'))
		self.PE_xbar_list = []
		self.PE_xbar_enable = []
		for i in range(self.group_num):
			self.PE_xbar_list.append([])
			self.PE_xbar_enable.append([])
			for j in range(self.PE_multiplex_xbar_num[0] * self.PE_multiplex_xbar_num[1]):
				__xbar = crossbar(SimConfig_path)
				self.PE_xbar_list[i].append(__xbar)
				self.PE_xbar_enable[i].append(0)

		self.PE_group_ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))*self.subarray_num
		self.PE_group_DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))*self.subarray_num
		self.PE_ADC_num = 0
		self.PE_DAC_num = 0

		self.input_demux = 0
		self.input_demux_power = 0
		self.input_demux_area = 0
		self.output_mux = 0
		self.output_mux_power = 0
		self.output_mux_area = 0

		self.calculate_ADC_num()
		self.calculate_DAC_num()

		self.PE_adder = adder(SimConfig_path)
		self.PE_adder_num = 0
		self.PE_shiftreg = shiftreg(SimConfig_path)
		self.PE_iReg = reg(SimConfig_path)
		self.PE_oReg = reg(SimConfig_path)

		self.PE_utilization = 0
		self.PE_max_occupied_column = 0

		self.PE_area = 0
		self.PE_xbar_area = 0
		self.PE_ADC_area = 0
		self.PE_DAC_area = 0
		self.PE_adder_area = 0
		self.PE_shiftreg_area = 0
		self.PE_iReg_area = 0
		self.PE_oReg_area = 0
		self.PE_input_demux_area = 0
		self.PE_output_mux_area = 0
		self.PE_digital_area = 0
		self.PE_inbuf_area = 0

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0
		self.PE_inbuf_read_rpower = 0
		self.PE_inbuf_read_wpower = 0
		self.PE_inbuf_read_power = 0

		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_adder_write_power = 0
		self.PE_shiftreg_write_power = 0
		self.PE_iReg_write_power = 0
		self.input_demux_write_power = 0
		self.output_mux_write_power = 0
		self.PE_digital_write_power = 0

		self.PE_read_latency = 0
		self.PE_xbar_read_latency = 0
		self.PE_ADC_read_latency = 0
		self.PE_DAC_read_latency = 0
		self.PE_adder_read_latency = 0
		self.PE_shiftreg_read_latency = 0
		self.PE_iReg_read_latency = 0
		self.input_demux_read_latency = 0
		self.output_mux_read_latency = 0
		self.PE_digital_read_latency = 0

		self.PE_write_latency = 0
		self.PE_xbar_write_latency = 0
		self.PE_ADC_write_latency = 0
		self.PE_DAC_write_latency = 0
		self.PE_adder_write_latency = 0
		self.PE_shiftreg_write_latency = 0
		self.PE_iReg_write_latency = 0
		self.input_demux_write_latency = 0
		self.output_mux_write_latency = 0
		self.PE_digital_write_latency = 0

		self.PE_read_energy = 0
		self.PE_xbar_read_energy = 0
		self.PE_ADC_read_energy = 0
		self.PE_DAC_read_energy = 0
		self.PE_adder_read_energy = 0
		self.PE_shiftreg_read_energy = 0
		self.PE_iReg_read_energy = 0
		self.input_demux_read_energy = 0
		self.output_mux_read_energy = 0
		self.PE_digital_read_energy = 0

		self.PE_write_energy = 0
		self.PE_xbar_write_energy = 0
		self.PE_ADC_write_energy = 0
		self.PE_DAC_write_energy = 0
		self.PE_adder_write_energy = 0
		self.PE_shiftreg_write_energy = 0
		self.PE_iReg_write_energy = 0
		self.input_demux_write_energy = 0
		self.output_mux_write_energy = 0
		self.PE_digital_write_energy = 0

		self.equ_power = 0
		self.equ_energy_efficiency = 0

		self.calculate_inter_PE_connection()

	def calculate_ADC_num(self):
		self.calculate_xbar_area()
		self.calculate_ADC_area()
		if self.PE_group_ADC_num == 0:
			self.PE_group_ADC_num = min((self.sub_position+1) * math.ceil(math.sqrt(self.xbar_area)*self.PE_multiplex_xbar_num[1]/math.sqrt(self.ADC_area)), self.xbar_column)*self.subarray_num
		else:
			assert self.PE_group_ADC_num > 0, "ADC number in one group < 0"
		self.PE_ADC_num = self.group_num * self.PE_group_ADC_num
		# self.output_mux = math.ceil(self.xbar_column*self.PE_multiplex_xbar_num[1]/self.PE_group_ADC_num)
		self.output_mux = math.ceil(self.xbar_column/(self.PE_group_ADC_num/self.subarray_num) * (self.sub_position+1))
		assert self.output_mux > 0

	def calculate_DAC_num(self):
		self.calculate_xbar_area()
		self.calculate_DAC_area()
		if self.PE_group_DAC_num == 0:
			self.PE_group_DAC_num = min(math.ceil(math.sqrt(self.xbar_area)/subarray_num * self.PE_multiplex_xbar_num[0] / math.sqrt(self.DAC_area)), self.subarray_size)*self.subarray_num
		else:
			assert self.PE_group_DAC_num > 0, "DAC number in one group < 0"
		self.PE_DAC_num = self.group_num * self.PE_group_DAC_num
		self.input_demux = math.ceil(self.subarray_size*self.PE_multiplex_xbar_num[0]/(self.PE_group_DAC_num/self.subarray_num))
		assert self.input_demux > 0

	def calculate_demux_area(self):
		transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
		demux_area_dict = {2: 8*transistor_area, # 2-1: 8 transistors
						   4: 24*transistor_area, # 4-1: 3 * 2-1
						   8: 72*transistor_area,
						   16: 216*transistor_area,
						   32: 648*transistor_area,
						   64: 1944*transistor_area
		}
		# unit: um^2
		# TODO: add circuits simulation results
		if self.input_demux <= 2:
			self.input_demux_area = demux_area_dict[2]
		elif self.input_demux<=4:
			self.input_demux_area = demux_area_dict[4]
		elif self.input_demux<=8:
			self.input_demux_area = demux_area_dict[8]
		elif self.input_demux<=16:
			self.input_demux_area = demux_area_dict[16]
		elif self.input_demux<=32:
			self.input_demux_area = demux_area_dict[32]
		else:
			self.input_demux_area = demux_area_dict[64]

	def calculate_mux_area(self):
		transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
		mux_area_dict = {2: 8*transistor_area,
						 4: 24*transistor_area,
						 8: 72*transistor_area,
						 16: 216*transistor_area,
						 32: 648*transistor_area,
						 64: 1944*transistor_area
		}
		# unit: um^2
		# TODO: add circuits simulation results
		if self.output_mux <= 2:
			self.output_mux_area = mux_area_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_area = mux_area_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_area = mux_area_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_area = mux_area_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_area = mux_area_dict[32]
		else:
			self.output_mux_area = mux_area_dict[64]

	def calculate_inter_PE_connection(self):
		temp = self.group_num
		self.PE_adder_num = 0
		while temp/2 >= 1:
			self.PE_adder_num += int(temp/2)*self.subarray_num
			temp = int(temp/2) + temp%2

	def PE_read_config(self, read_row = None, read_column = None, read_matrix = None, read_vector = None):
		# read_row and read_column are lists with the length of #occupied groups
		# read_matrix is a 2D list of matrices. The size of the list is (#occupied groups x Xbar_Polarity)
		# read_vector is a list of vectors with the length of #occupied groups
		self.PE_utilization = 0
		if self.PE_simulation_level == 0:
			if (read_row is None) or (read_column is None):
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_read_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
			else:
				assert len(read_row) == len(read_column), "read_row and read_column must be equal in length"
				self.num_occupied_group = len(read_row)
				assert self.num_occupied_group <= self.group_num, "The length of read_row exceeds the group number in one PE"
				for i in range(self.group_num):
					if i < self.num_occupied_group:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_list[i][0].xbar_read_config(read_row = read_row[i], read_column = read_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						else:
							self.PE_xbar_list[i][0].xbar_read_config(read_row=read_row[i], read_column=read_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							self.PE_xbar_list[i][1].xbar_read_config(read_row=read_row[i], read_column=read_column[i])
							self.PE_xbar_enable[i][1] = 1
							self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
					else:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_enable[i][0] = 0
						else:
							self.PE_xbar_enable[i][0] = 0
							self.PE_xbar_enable[i][1] = 0
		else:
			if read_matrix is None:
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_read_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				if read_vector is None:
					self.num_occupied_group = len(read_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of read_matrix exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix=read_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix=read_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_read_config(read_matrix=read_matrix[i][1])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
				else:
					assert len(read_matrix) == len(read_vector), "The number of read_matrix and read_vector must be equal"
					self.num_occupied_group = len(read_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of read_matrix/read_vector exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix = read_matrix[i][0], read_vector = read_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix = read_matrix[i][0], read_vector = read_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_read_config(read_matrix = read_matrix[i][1], read_vector = read_vector[i])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
		self.PE_utilization /= (self.group_num * self.PE_multiplex_xbar_num[1])

	def PE_write_config(self, write_row=None, write_column=None, write_matrix=None, write_vector=None):
		# write_row and write_column are array with the length of #occupied groups
		# write_matrix is a 2D array of matrices. The size of the list is (#occupied groups x Xbar_Polarity)
		# write_vector is a array of vector with the length of #occupied groups
		if self.PE_simulation_level == 0:
			if (write_row is None) or (write_column is None):
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_write_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				assert len(write_row) == len(write_column), "write_row and write_column must be equal in length"
				self.num_occupied_group = len(write_row)
				assert self.num_occupied_group <= self.group_num, "The length of write_row exceeds the group number in one PE"
				for i in range(self.group_num):
					if i < self.num_occupied_group:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_list[i][0].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						else:
							self.PE_xbar_list[i][0].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							self.PE_xbar_list[i][1].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][1] = 1
							self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
					else:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_enable[i][0] = 0
						else:
							self.PE_xbar_enable[i][0] = 0
							self.PE_xbar_enable[i][1] = 0
		else:
			if write_matrix is None:
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_write_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				if write_vector is None:
					self.num_occupied_group = len(write_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of write_matrix exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_write_config(write_matrix=write_matrix[i][1])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
				else:
					assert len(write_matrix) == len(write_vector), "The number of write_matrix and write_vector must be equal"
					self.num_occupied_group = len(write_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of write_matrix/write_vector exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_write_config(write_matrix=write_matrix[i][1],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
		self.PE_utilization /= (self.group_num * self.PE_multiplex_xbar_num[1])

	def calculate_PE_area(self, SimConfig_path=None, default_inbuf_size = 16):
		# unit: um^2
		self.inbuf = buffer(SimConfig_path=SimConfig_path,buf_level=1,default_buf_size=default_inbuf_size)
		self.inbuf.calculate_buf_area()
		self.calculate_xbar_area()
		self.calculate_demux_area()
		self.calculate_mux_area()
		self.calculate_DAC_area()
		self.calculate_ADC_area()
		self.PE_adder.calculate_adder_area()
		self.PE_shiftreg.calculate_shiftreg_area()
		self.PE_iReg.calculate_reg_area()
		self.PE_oReg.calculate_reg_area()
		self.PE_xbar_area = self.PE_xbar_num*self.xbar_area
		self.PE_ADC_area = self.ADC_area*self.PE_ADC_num
		self.PE_DAC_area = self.DAC_area*self.PE_DAC_num
		self.PE_adder_area = self.PE_group_ADC_num*self.PE_adder_num*self.PE_adder.adder_area
		self.PE_shiftreg_area = self.PE_ADC_num*self.PE_shiftreg.shiftreg_area
		self.PE_iReg_area = self.PE_DAC_num*self.PE_iReg.reg_area
		self.PE_oReg_area = self.PE_ADC_num*self.PE_oReg.reg_area
		self.PE_input_demux_area = self.input_demux_area*self.PE_DAC_num
		self.PE_output_mux_area = self.output_mux_area*self.PE_ADC_num
		self.PE_digital_area = self.PE_adder_area + self.PE_shiftreg_area + self.PE_input_demux_area + self.PE_output_mux_area + self.PE_iReg_area + self.PE_oReg_area
		self.PE_inbuf_area = self.inbuf.buf_area
		self.PE_area = self.PE_xbar_area + self.PE_ADC_area + self.PE_DAC_area + self.PE_digital_area + self.PE_inbuf_area
		
	
	def calculate_demux_power(self):
		transistor_power = 10*1.2/1e9
		demux_power_dict = {2: 8*transistor_power,
						 4: 24*transistor_power,
						 8: 72*transistor_power,
						 16: 216*transistor_power,
						 32: 648*transistor_power,
						 64: 1944*transistor_power
		}
		# unit: W
		# TODO: add circuits simulation results
		if self.input_demux <= 2:
			self.input_demux_power = demux_power_dict[2]
		elif self.input_demux<=4:
			self.input_demux_power = demux_power_dict[4]
		elif self.input_demux<=8:
			self.input_demux_power = demux_power_dict[8]
		elif self.input_demux<=16:
			self.input_demux_power = demux_power_dict[16]
		elif self.input_demux<=32:
			self.input_demux_power = demux_power_dict[32]
		else:
			self.input_demux_power = demux_power_dict[64]

	def calculate_mux_power(self):
		transistor_power = 10*1.2/1e9
		mux_power_dict = {2: 8*transistor_power,
						 4: 24*transistor_power,
						 8: 72*transistor_power,
						 16: 216*transistor_power,
						 32: 648*transistor_power,
						 64: 1944*transistor_power
		}
		# unit: W
		# TODO: add circuits simulation results
		if self.output_mux <= 2:
			self.output_mux_power = mux_power_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_power = mux_power_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_power = mux_power_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_power = mux_power_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_power = mux_power_dict[32]
		else:
			self.output_mux_power = mux_power_dict[64]

	def calculate_PE_read_power_fast(self, max_column=0, max_row=0, max_group=0, SimConfig_path=None, default_inbuf_size = 16):
		# unit: W
		# coarse but fast estimation
		# max_column: maximum used column in one crossbar in this tile
		# max_row: maximum used row in one crossbar in this tile
		# max_group: maximum used groups in one PE
		self.inbuf = buffer(SimConfig_path=SimConfig_path, buf_level=1, default_buf_size=default_inbuf_size)
		self.inbuf.calculate_buf_read_power()
		self.inbuf.calculate_buf_write_power()
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_shiftreg.calculate_shiftreg_power()
		self.PE_iReg.calculate_reg_power()
		self.PE_oReg.calculate_reg_power()
		self.PE_adder.calculate_adder_power()
		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0
		
		
		self.xbar_read_config(read_row=max_row, read_column=max_column)
		self.calculate_xbar_read_power()
		self.PE_xbar_read_power = self.PE_multiplex_xbar_num[1]*max_group*self.xbar_read_power/self.input_demux/self.output_mux
		self.PE_DAC_read_power = max_group*math.ceil(max_row/self.input_demux) * self.DAC_power
		self.input_demux_read_power = max_group*math.ceil(max_row/self.input_demux) * self.input_demux_power
		self.PE_iReg_read_power = max_group * math.ceil(max_row/self.input_demux) * self.PE_iReg.reg_power

		if self.PIM_type_pe == 0: # analog pim
			self.PE_ADC_read_power = max_group*math.ceil(max_column/self.output_mux) * self.ADC_power
			self.output_mux_read_power = max_group*math.ceil(max_column/self.output_mux) * self.output_mux_power
			self.PE_adder_read_power = (max_group - 1) * math.ceil(max_column/self.output_mux) * self.PE_adder.adder_power
			self.PE_shiftreg_read_power = max_group * math.ceil(max_column/self.output_mux) * self.PE_shiftreg.shiftreg_power
			self.PE_oReg_read_power = max_group * math.ceil(max_column / self.output_mux) * self.PE_oReg.reg_power
		else: # digital pim
			self.PE_ADC_read_power = max_group*math.ceil(max_column/self.output_mux) * self.ADC_power * self.subarray_num
			self.output_mux_read_power = max_group*math.ceil(max_column/self.output_mux) * self.output_mux_power * self.subarray_num
			self.PE_adder_read_power = (max_group - 1) * math.ceil(max_column/self.output_mux) * self.PE_adder.adder_power * self.subarray_num
			self.PE_shiftreg_read_power = max_group * math.ceil(max_column/self.output_mux) * self.PE_shiftreg.shiftreg_power * self.subarray_num
			self.PE_oReg_read_power = max_group * math.ceil(max_column / self.output_mux) * self.PE_oReg.reg_power * self.subarray_num

		self.PE_digital_read_power = self.input_demux_read_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power + self.PE_iReg_read_power + self.PE_oReg_read_power
		self.PE_inbuf_read_rpower = self.inbuf.buf_rpower*1e-3
		self.PE_inbuf_read_wpower = self.inbuf.buf_wpower * 1e-3
		self.PE_inbuf_read_power = self.PE_inbuf_read_rpower + self.PE_inbuf_read_wpower
		self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power + self.PE_digital_read_power + self.PE_inbuf_read_power

	def calculate_PE_read_power(self):
		# unit: W
		# Notice: before calculating latency, PE_read_config must be executed
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_shiftreg.calculate_shiftreg_power()
		self.PE_iReg.calculate_reg_power()
		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0
		self.PE_max_occupied_column = 0
		if self.num_occupied_group != 0:
			for i in range(self.group_num):
				if self.PE_xbar_enable[i][0] == 1:
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][0].xbar_read_power/self.input_demux/self.output_mux
					else:
						self.PE_xbar_list[i][0].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][0].xbar_read_power/self.input_demux/self.output_mux
						self.PE_xbar_list[i][1].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][1].xbar_read_power/self.input_demux/self.output_mux
					self.PE_DAC_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.input_demux)*self.DAC_power
					self.PE_iReg_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.input_demux)*self.PE_iReg.shiftreg_power
					self.input_demux_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.input_demux)*self.input_demux_power
					if self.PIM_type_pe == 0:
						self.PE_ADC_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.ADC_power
						self.output_mux_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.output_mux_power
					else:
						self.PE_ADC_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.ADC_power*self.subarray_num
						self.output_mux_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.output_mux_power*self.subarray_num
					if self.PE_xbar_list[i][0].xbar_num_read_column > self.PE_max_occupied_column:
						# find the most occupied column of each group in one PE
						self.PE_max_occupied_column = self.PE_xbar_list[i][0].xbar_num_read_column
			# PE_max_read_column = min(self.PE_max_occupied_column, self.PE_group_ADC_num)
			if self.PIM_type_pe == 0:
				self.PE_adder_read_power = (self.num_occupied_group-1)*(self.PE_max_occupied_column/self.output_mux)*self.PE_adder.adder_power
				self.PE_shiftreg_read_power = (self.num_occupied_group)*(self.PE_max_occupied_column/self.output_mux)*self.PE_shiftreg.shiftreg_power
			else:
				self.PE_adder_read_power = (self.num_occupied_group-1)*(self.PE_max_occupied_column/self.output_mux)*self.PE_adder.adder_power*self.subarray_num
				self.PE_shiftreg_read_power = (self.num_occupied_group)*(self.PE_max_occupied_column/self.output_mux)*self.PE_shiftreg.shiftreg_power*self.subarray_num
			self.PE_digital_read_power = self.input_demux_read_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power + self.PE_iReg_read_power
			self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power + self.PE_digital_read_power

	def calculate_PE_energy_efficiency(self, SimConfig_path=None):
		PE_config = cp.ConfigParser()
		PE_config.read(SimConfig_path, encoding='UTF-8')
		self.digital_period = 1/float(PE_config.get('Digital module', 'Digital_Frequency'))*1e3
		multiple_time = math.ceil(8/self.DAC_precision)
		decoder1_8 = 0.27933
		Row_per_DAC = math.ceil(self.subarray_size/(self.PE_group_DAC_num/self.subarray_num))
		m = 1
		while Row_per_DAC>0:
			Row_per_DAC = Row_per_DAC // 8
			m += 1
		self.decoderLatency = m * decoder1_8
		
		mux8_1 = 32.744*1e-3
		m = 1
		Column_per_ADC = math.ceil(self.xbar_size[1] / (self.PE_group_ADC_num/self.subarray_num))
		while Column_per_ADC > 0:
			Column_per_ADC = Column_per_ADC // 8
			m += 1
		self.muxLatency = m * mux8_1
		# print("multiple:", multiple_time)
		self.calculate_xbar_read_latency()
		self.calculate_DAC_latency()
		self.calculate_ADC_latency()
		xbar_latency = multiple_time * self.xbar_read_latency
		# print("xbar lat:", xbar_latency)
		DAC_latency = multiple_time * self.DAC_latency
		ADC_latency = multiple_time * self.ADC_latency
		iReg_latency = self.digital_period+multiple_time*self.digital_period
		shiftreg_latency =  multiple_time * self.digital_period
		input_demux_latency = multiple_time*self.decoderLatency
		adder_latency = math.ceil(math.log2(self.group_num))*self.digital_period
		output_mux_latency = multiple_time*self.muxLatency
		oReg_latency = self.digital_period

		total_latency = xbar_latency+DAC_latency+ADC_latency+iReg_latency+input_demux_latency+shiftreg_latency+adder_latency+output_mux_latency+oReg_latency
		print("xbar_lat", xbar_latency)
		print("ADC lat", ADC_latency)
		print(DAC_latency)
		print("lat", total_latency)

		total_ops = 2*self.PE_group_DAC_num * self.PE_group_ADC_num
		print(self.PE_group_DAC_num)
		print(self.PE_group_ADC_num)
		print(total_ops)


		xbar_energy = xbar_latency * self.PE_xbar_read_power
		DAC_energy = DAC_latency * self.PE_DAC_read_power
		ADC_energy = ADC_latency * self.PE_ADC_read_power
		iReg_energy = iReg_latency * self.PE_iReg_read_power
		shiftreg_energy = shiftreg_latency * self.PE_shiftreg_read_power
		input_demux_energy = input_demux_latency * self.input_demux_read_power
		adder_energy = adder_latency * self.PE_adder_read_power
		output_mux_energy = output_mux_latency * self.output_mux_read_power
		oReg_energy = oReg_latency * self.PE_oReg_read_power

		total_energy = xbar_energy+DAC_energy+ADC_energy+iReg_energy+shiftreg_energy+input_demux_energy+adder_energy+output_mux_energy+oReg_energy
		print("xbar_area:", self.PE_xbar_area)
		print("macro_area:", self.PE_xbar_area+self.PE_ADC_area+self.PE_DAC_area+self.PE_digital_area)
		print("xbar_energy", xbar_energy)
		print("DAC_energy", DAC_energy)
		print("ADC_power", self.PE_ADC_read_power)
		print("ADC_latenct", self.ADC_latency)
		print("ADC_energy", ADC_energy)

		self.equ_power = total_ops/total_latency #unit: GOPS
		self.equ_energy_efficiency = total_ops/total_energy #unit: GOPS/W






	'''def calculate_PE_write_power(self):
		# unit: W
		# Notice: before calculating latency, PE_write_config must be executed
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_adder_write_power = 0
		self.PE_shiftreg_write_power = 0
		self.input_demux_write_power = 0
		self.output_mux_write_power = 0
		self.PE_digital_write_power = 0
		if self.num_occupied_group != 0:
			for i in range(self.group_num):
				if self.PE_xbar_enable[i][0] == 1:
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][0].xbar_write_power
					else:
						self.PE_xbar_list[i][0].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][0].xbar_write_power
						self.PE_xbar_list[i][1].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][1].xbar_write_power
					self.PE_DAC_write_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_write_row/self.input_demux)*self.DAC_power
					self.PE_ADC_write_power += 0
					# Assume ADCs are idle in write process
					self.input_demux_write_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_write_row/self.input_demux)*self.input_demux_power
			self.PE_digital_write_power = self.input_demux_write_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power
			self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_ADC_write_power + self.PE_digital_write_power

	def calculate_PE_read_energy(self):
		# unit: nJ
		# Notice: before calculating energy, PE_read_config and calculate_PE_read_power must be executed
		self.PE_xbar_read_energy = self.PE_xbar_read_latency * self.PE_xbar_read_power
		self.PE_DAC_read_energy = self.PE_DAC_read_latency * self.PE_DAC_read_power
		self.PE_ADC_read_energy = self.PE_ADC_read_latency * self.PE_ADC_read_power
		self.PE_adder_read_energy = self.PE_adder_read_power * self.PE_adder_read_latency
		self.PE_shiftreg_read_energy = self.PE_shiftreg_read_power * self.PE_shiftreg_read_latency
		self.input_demux_read_energy = self.input_demux_read_power * self.input_demux_read_latency
		self.output_mux_read_energy = self.output_mux_read_power * self.output_mux_read_latency
		self.PE_digital_read_energy = self.PE_adder_read_energy + self.PE_shiftreg_read_energy + self.input_demux_read_energy + self.output_mux_read_energy
		self.PE_read_energy = self.PE_xbar_read_energy + self.PE_DAC_read_energy + \
							  self.PE_ADC_read_energy + self.PE_digital_read_energy

	def calculate_PE_write_energy(self):
		# unit: nJ
		# Notice: before calculating energy, PE_write_config and calculate_PE_write_power must be executed
		self.PE_xbar_write_energy = self.PE_xbar_write_latency * self.PE_xbar_write_power
		self.PE_DAC_write_energy = self.PE_DAC_write_latency * self.PE_DAC_write_power
		self.PE_ADC_write_energy = self.PE_ADC_write_latency * self.PE_ADC_write_power
		self.PE_adder_write_energy = self.PE_adder_write_power * self.PE_adder_write_latency
		self.PE_shiftreg_write_energy = self.PE_shiftreg_write_power * self.PE_shiftreg_write_latency
		self.input_demux_write_energy = self.input_demux_write_power * self.input_demux_write_latency
		self.output_mux_write_energy = self.output_mux_write_latency * self.output_mux_write_power
		self.PE_digital_write_energy = self.PE_adder_write_energy + self.PE_shiftreg_write_energy + self.input_demux_write_energy + self.output_mux_write_energy
		self.PE_write_energy = self.PE_xbar_write_energy + self.PE_DAC_write_energy + \
							  self.PE_ADC_write_energy + self.PE_digital_write_energy'''

	def PE_output(self):
		print("---------------------Crossbar Configurations-----------------------")
		crossbar.xbar_output(self)
		print("------------------------DAC Configurations-------------------------")
		DAC.DAC_output(self)
		print("------------------------ADC Configurations-------------------------")
		ADC.ADC_output(self)
		print("-------------------------PE Configurations-------------------------")
		print("total crossbar number in one PE:", self.PE_xbar_num)
		print("			the number of crossbars sharing a set of interfaces:",self.PE_multiplex_xbar_num)
		print("total utilization rate:", self.PE_utilization)
		print("total DAC number in one PE:", self.PE_DAC_num)
		print("			the number of DAC in one set of interfaces:", self.PE_group_DAC_num)
		print("total ADC number in one PE:", self.PE_ADC_num)
		print("			the number of ADC in one set of interfaces:", self.PE_group_ADC_num)
		print("---------------------PE Area Simulation Results--------------------")
		print("PE area:", self.PE_area, "um^2")
		print("			crossbar area:", self.PE_xbar_area, "um^2")
		print("			DAC area:", self.PE_DAC_area, "um^2")
		print("			ADC area:", self.PE_ADC_area, "um^2")
		print("			digital part area:", self.PE_digital_area, "um^2")
		print("			|---adder area:", self.PE_adder_area, "um^2")
		print("			|---shift-reg area:", self.PE_shiftreg_area, "um^2")
		print("			|---input_demux area:", self.PE_input_demux_area, "um^2")
		print("			|---output_mux area:", self.PE_output_mux_area, "um^2")
		print("--------------------PE Latency Simulation Results-----------------")
		print("PE read latency:", self.PE_read_latency, "ns")
		print("			crossbar read latency:", self.PE_xbar_read_latency, "ns")
		print("			DAC read latency:", self.PE_DAC_read_latency, "ns")
		print("			ADC read latency:", self.PE_ADC_read_latency, "ns")
		print("			digital part read latency:", self.PE_digital_read_latency, "ns")
		print("PE write latency:", self.PE_write_latency, "ns")
		print("			crossbar write latency:", self.PE_xbar_write_latency, "ns")
		print("			DAC write latency:", self.PE_DAC_write_latency, "ns")
		print("			ADC write latency:", self.PE_ADC_write_latency, "ns")
		print("			digital part write latency:", self.PE_digital_write_latency, "ns")
		print("--------------------PE Power Simulation Results-------------------")
		print("PE read power:", self.PE_read_power, "W")
		print("			crossbar read power:", self.PE_xbar_read_power, "W")
		print("			DAC read power:", self.PE_DAC_read_power, "W")
		print("			ADC read power:", self.PE_ADC_read_power, "W")
		print("			digital part read power:", self.PE_digital_read_power, "W")
		print("			|---adder power:", self.PE_adder_read_power, "W")
		print("			|---shift-reg power:", self.PE_shiftreg_read_power, "W")
		print("			|---input_demux power:", self.input_demux_read_power, "W")
		print("			|---output_mux power:", self.output_mux_read_power, "W")
		print("PE write power:", self.PE_write_power, "W")
		print("			crossbar write power:", self.PE_xbar_write_power, "W")
		print("			DAC write power:", self.PE_DAC_write_power, "W")
		print("			ADC write power:", self.PE_ADC_write_power, "W")
		print("			digital part write power:", self.PE_digital_write_power, "W")
		print("------------------PE Energy Simulation Results--------------------")
		print("PE read energy:", self.PE_read_energy, "nJ")
		print("			crossbar read energy:", self.PE_xbar_read_energy, "nJ")
		print("			DAC read energy:", self.PE_DAC_read_energy, "nJ")
		print("			ADC read energy:", self.PE_ADC_read_energy, "nJ")
		print("			digital part read energy:", self.PE_digital_read_energy, "nJ")
		print("PE write energy:", self.PE_write_energy, "nJ")
		print("			crossbar write energy:", self.PE_xbar_write_energy, "nJ")
		print("			DAC write energy:", self.PE_DAC_write_energy, "nJ")
		print("			ADC write energy:", self.PE_ADC_write_energy, "nJ")
		print("			digital part write energy:", self.PE_digital_write_energy, "nJ")
		print("-----------------------------------------------------------------")
	
def PE_test():
	print("load file:",test_SimConfig_path)
	_PE = ProcessElement(test_SimConfig_path)
	_PE.calculate_PE_area(test_SimConfig_path)
	_PE.calculate_PE_read_power_fast(_PE.xbar_size[1], _PE.xbar_size[0], _PE.group_num, test_SimConfig_path)
	_PE.calculate_PE_energy_efficiency(test_SimConfig_path)
	# _PE.calculate_PE_read_energy()
	# _PE.PE_output()
	print(_PE.PE_area)
	print(_PE.equ_power)
	print(_PE.equ_energy_efficiency)


if __name__ == '__main__':
	PE_test()