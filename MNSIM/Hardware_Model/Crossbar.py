#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
from MNSIM.Hardware_Model.Device import device
import numpy as np
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class crossbar(device):
	def __init__(self, SimConfig_path):
		device.__init__(self,SimConfig_path)
		xbar_config = cp.ConfigParser()
		xbar_config.read(SimConfig_path, encoding='UTF-8')
		self.xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
		self.xbar_row = int(self.xbar_size[0])
		self.xbar_column = int(self.xbar_size[1])
		self.subarray_size = int(xbar_config.get('Crossbar level', 'Subarray_Size'))
		assert self.xbar_row % self.subarray_size == 0, "The crossbar size must be divisible by the subarray size"
		self.subarray_num = self.xbar_row/self.subarray_size
		
		self.cell_type = xbar_config.get('Crossbar level', 'Cell_Type')
		self.transistor_tech = int(xbar_config.get('Crossbar level', 'Transistor_Tech'))
		self.wire_resistance = float(xbar_config.get('Crossbar level', 'Wire_Resistance'))
		self.wire_capacity = float(xbar_config.get('Crossbar level', 'Wire_Capacity'))
		self.area_calculation_method = int(xbar_config.get('Crossbar level', 'Area_Calculation'))
		self.xbar_area = 0
		self.xbar_simulation_level = int(xbar_config.get('Algorithm Configuration', 'Simulation_Level'))

		if self.device_type == 'NVM':
			self.xbar_load_resistance = float(xbar_config.get('Crossbar level', 'Load_Resistance'))
			if self.xbar_load_resistance == -1:
				self.xbar_load_resistance = math.sqrt(self.device_resistance[0] * self.device_resistance[-1])
			assert self.xbar_load_resistance >0, "Load resistance must be > 0"

		self.xbar_write_matrix = np.zeros((self.xbar_row,self.xbar_column))# * 1/self.device_resistance[-1]
		self.xbar_write_vector = np.zeros((self.xbar_row,1))
		self.xbar_read_matrix = np.zeros((self.xbar_row, self.xbar_column))# * 1/self.device_resistance[-1]
		self.xbar_read_vector = np.zeros((self.xbar_row, 1))

		self.xbar_read_power = 0
		self.xbar_read_latency = 0

		self.xbar_write_power = 0
		self.xbar_write_latency = 0

		self.xbar_read_energy = 0
		self.xbar_write_energy = 0

		self.xbar_num_write_row = 0
		self.xbar_num_write_column = 0

		self.xbar_num_read_row = 0
		self.xbar_num_read_column = 0

		self.xbar_utilization = 0
		# print("Crossbar configuration is loaded")


	def xbar_write_config(self, write_row = None, write_column = None, write_matrix = None, write_vector = None):
		# write_row and write_column are the sizes of occupied parts in crossbars
		# write_matrix: the target weight matrix of the write operation, write_vector: the vector of write voltages
		if self.xbar_simulation_level == 0:
		# behavior level simulation
			if write_row is None:
				self.xbar_num_write_row = self.xbar_row
			else:
				assert write_row >= 0, "Num of occupied row (write) < 0"
				self.xbar_num_write_row = write_row
			if write_column is None:
				self.xbar_num_write_column = self.xbar_column
			else:
				assert write_column >= 0, "Num of occupied column (write) < 0"
				self.xbar_num_write_column = write_column
		else:
		# estimation level simulation
			if write_matrix is None or len(write_matrix) == 0 or ((len(write_matrix)>0) & (len(write_matrix[0])==0)):
				self.xbar_write_matrix = 1/math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1])) \
				  * np.ones((self.xbar_row,self.xbar_column))
				self.xbar_num_write_column = self.xbar_column
				self.xbar_num_write_row = self.xbar_row
			else:
				for i in range(len(write_matrix)):
					for j in range(len(write_matrix[0])):
						assert int(write_matrix[i][j]) < self.device_level, "Weight value (write) exceeds the resistance range"
						self.xbar_write_matrix[i][j] = 1/self.device_resistance[int(write_matrix[i][j])]
				self.xbar_num_write_row = len(write_matrix)
				self.xbar_num_write_column = len(write_matrix[0])
			if write_vector is None or len(write_vector) == 0 or ((len(write_vector)>0) & (len(write_vector[0])==0)):
				self.xbar_write_vector = math.sqrt((self.device_write_voltage[0]*self.device_write_voltage[-1])) \
										 * np.ones((self.xbar_row,1))
			else:
				for i in range(len(write_vector)):
					assert int(write_vector[i][0]) < self.device_write_voltage_level, "Write voltage value is out of range"
					self.xbar_write_vector[i][0] = self.device_write_voltage[int(write_vector[i][0])]
		self.xbar_utilization = self.xbar_num_write_column * self.xbar_num_write_row / (self.xbar_row*self.xbar_column)

	def xbar_read_config(self, read_row = None, read_column = None, read_matrix = None, read_vector = None):
		# read_row and read_column are the sizes of occupied parts in crossbars
		# read_matrix: the weight matrix stored in the crossbar, read_vector: read voltage vector (activation input)
		if self.xbar_simulation_level == 0:
		# behavior level simulation
			if read_row is None:
				self.xbar_num_read_row = self.xbar_row
			else:
				assert read_row >= 0, "Num of occupied row (read) < 0"
				self.xbar_num_read_row = read_row
			if read_column is None:
				self.xbar_num_read_column = self.xbar_column
			else:
				assert read_column >= 0, "Num of occupied column (read) < 0"
				self.xbar_num_read_column = read_column
		else:
		# estimation level simulation
			if read_matrix is None or len(read_matrix) == 0 or ((len(read_matrix)>0) & (len(read_matrix[0])==0)):
				self.xbar_read_matrix = 1/math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1])) \
										 * np.ones((self.xbar_row,self.xbar_column))
				self.xbar_num_read_column = self.xbar_column
				self.xbar_num_read_row = self.xbar_row
			else:
				for i in range(len(read_matrix)):
					for j in range(len(read_matrix[0])):
						assert int(read_matrix[i][j]) < self.device_level, "Weight value (read) exceeds the resistance range"
						self.xbar_read_matrix[i][j] = 1/self.device_resistance[int(read_matrix[i][j])]
				self.xbar_num_read_row = len(read_matrix)
				self.xbar_num_read_column = len(read_matrix[0])
			if read_vector is None or len(read_vector) == 0 or ((len(read_vector)>0) & (len(read_vector[0])==0)):
				# self.xbar_read_vector = math.sqrt((self.device_read_voltage[0]*self.device_read_voltage[-1])) \
				# 						 * np.ones((self.xbar_row,1))
				self.xbar_read_vector = math.sqrt((self.device_read_voltage[0]**2 + self.device_read_voltage[-1]**2)/2) * np.ones((self.xbar_row,1))
				# self.xbar_read_vector = self.device_read_voltage[-1] * np.ones((self.xbar_row,1))
			else:
				for i in range(len(read_vector)):
					assert int(read_vector[i][0]) < self.device_read_voltage_level, "Vector value exceeds the input voltage range"
					self.xbar_read_vector[i][0] = self.device_read_voltage[int(read_vector[i][0])]
		self.xbar_utilization = self.xbar_num_read_row * self.xbar_num_read_column / (self.xbar_row * self.xbar_column)

	def calculate_xbar_area(self):
		# Area unit: um^2
		if self.area_calculation_method == 0:
			area_factor = 1
			# self.xbar_area = area_factor * self.xbar_row * self.xbar_column * self.device_area + self.xbar_row * (7.3*2.7+9.5*3.8)*(self.transistor_tech/65)**2
			self.xbar_area = area_factor * self.xbar_row * self.xbar_column * self.device_area + 1150*self.xbar_row
			# when consider driver, the area is: self.xbar_row * (7.3*2.7+9.5*3.8)*(self.transistor_tech/65)**2
			# in A Fully Integrated Analog ReRAM Based 78.4TOPS/W Compute-In-Memory Chip with Fully Parallel MAC Computing, the driver area is 1150, for others, the driver area is (7.3*2.7+9.5*3.8)*(self.transistor_tech/65)**2, ref: Multi-level wordline driver for robust SRAM design in nano-scale CMOS technology
		else:
			if self.device_type == 'SRAM':
				self.xbar_area = self.xbar_row * self.xbar_column * 5
			else:
				WL_ratio = 3
				# WL_ratio is the technology parameter W/L of the transistor
				if self.cell_type[0] == '0':
					self.xbar_area = 4 * self.xbar_row * self.xbar_column * self.device_tech**2 * 1e-6
				else:
					self.xbar_area = 3 * (WL_ratio + 1) * self.xbar_row * self.xbar_column * self.device_tech**2 * 1e-6


	def calculate_wire_resistance(self):
		#unit: ohm
		if self.wire_resistance < 0:
			self.wire_resistance = 2.82
				#ref: Overcoming the challenges of crossbar resistive memory architectures
				#TODO: Update the wire resistance calculation according to different technology sizes

	def calculate_wire_capacity(self):
		#unit: fF
		if self.wire_capacity < 0:
			self.wire_capacity = 1
				#TODO: Update the wire capacity calculation according to different technology sizes

	def calculate_xbar_read_latency(self):
		# unit: ns
		self.calculate_wire_resistance()
		self.calculate_wire_capacity()
		if self.cell_type == '0T1R':
			size = self.xbar_row*self.xbar_column / 1024 / 8  # KB
			wire_latency = 0.001 * (0.0002 * size ** 2 + 5 * 10 ** -6 * size + 4 * 10 ** -14)  # ns
		else:
			size = self.xbar_row * self.xbar_column / 1024 / 8  # KB
			wire_latency = 0.001 * (0.0002 * size ** 2 + 5 * 10 ** -6 * size + 4 * 10 ** -14)  # ns
		# wire_latency = 0.5 * self.wire_resistance * self.wire_capacity * self.xbar_row 1e3
			#TODO: Update the calculation formula considering the branches
		self.xbar_read_latency = self.device_read_latency + wire_latency

	def calculate_xbar_write_latency(self):
		# Notice: before calculating write latency, xbar_write_config must be executed
		self.xbar_write_latency	= self.device_write_latency * self.xbar_num_write_row
		# self.xbar_write_latency = self.device_write_latency * min(math.ceil(num_write_row/num_multi_row), num_write_column)\
		#                           * min(num_multi_row, num_write_row)
			#unit: ns
			#Assuming that the write operation of cells in one row can be performed concurrently

	def calculate_xbar_read_power(self):
		# unit: W
		# cal_mode: 0: simple estimation, 1: detailed simulation
		# Notice: before calculating power, xbar_read_config must be executed
		# self.xbar_read_config(read_matrix,read_vector)
		self.xbar_read_power = 0
		#Assuming that in 0T1R structure, the read power of unselected cell is 1/4 of the selected cells' read power
		if self.xbar_simulation_level == 0:
			assert self.xbar_utilization <= 1, "Crossbar usage utilization rate > 1"
			self.calculate_device_read_power()
			self.xbar_read_power += self.xbar_num_read_row * self.xbar_num_read_column * self.device_read_power
			if self.cell_type[0] =='0':
				self.calculate_device_read_power(self.device_resistance[0])
				self.xbar_read_power += 0.25 * self.xbar_num_read_row * (self.xbar_column - self.xbar_num_read_column) \
										* self.device_read_power
		else:
			temp_v2 = self.xbar_read_vector * self.xbar_read_vector
			self.xbar_read_power += (self.xbar_read_matrix.T.dot(temp_v2)).sum()
			if self.device_type == 'NVM' and self.cell_type[0] == '0':
				temp_matrix = np.ones((self.xbar_num_read_row, self.xbar_column-self.xbar_num_read_column)) / self.device_resistance[0]
				# print("temp_matrix",temp_matrix.T)
				# print("temp_vector",temp_v2[0:self.xbar_num_read_column])
				# print("unused power", 0.25* (temp_matrix.T.dot(temp_v2[0:self.xbar_num_read_column])).sum())
				self.xbar_read_power += 0.25* (temp_matrix.T.dot(temp_v2[0:self.xbar_num_read_column])).sum()

	def calculate_xbar_write_power(self):
		# unit: W
		# cal_mode: 0: simple estimation, 1: detailed simulation
		# Notice: before calculating power, xbar_write_config must be executed
		# TODO: consider the write power of unselected cells

		# self.xbar_write_config(write_matrix, write_vector)
		self.xbar_write_power = 0
		# Assuming that the write operation of cells in one row can be performed concurrently
		if self.xbar_simulation_level == 0:
			self.calculate_device_write_power()
			self.xbar_write_power = self.xbar_num_write_column * self.device_write_power
		else:
			temp_v2 = self.xbar_write_vector * self.xbar_write_vector
			assert self.xbar_num_write_row>0, "xbar_num_write_row is 0, consider use the write_config function"
			self.xbar_write_power = (self.xbar_write_matrix.T.dot(temp_v2)).sum() / self.xbar_num_write_row

	def calculate_xbar_read_energy(self):
		#unit: nJ
		if self.device_type == 'NVM':
			self.xbar_read_energy = self.xbar_read_power * self.xbar_read_latency
		elif self.device_type == 'SRAM':
			self.xbar_read_energy = self.xbar_num_read_row * self.xbar_num_read_column * self.device_read_energy * 1e6

	def calculate_xbar_write_energy(self):
		#unit: nJ
		if self.device_type == 'NVM':
			self.xbar_write_energy = self.xbar_write_power * self.device_write_latency
			#Do not consider the wire power in write operation	
		elif self.device_type == 'SRAM':
			self.xbar_write_energy = self.xbar_num_write_row * self.xbar_num_write_column * self.device_write_energy * 1e6

	def xbar_output(self):
		device.device_output(self)
		print("crossbar_size:", self.xbar_size)
		print("cell_type:", self.cell_type)
		print("transistor_tech:", self.transistor_tech, "nm")
		print("wire_resistance:", self.wire_resistance, "ohm")
		print("wire_capacity:", self.wire_capacity, "fF")
		print("crossbar_area", self.xbar_area, "um^2")
		print("load_resistance", self.xbar_load_resistance, "ohm")
		print("crossbar_utilization_rate", self.xbar_utilization)
		print("crossbar_read_power:", self.xbar_read_power, "W")
		print("crossbar_read_latency:", self.xbar_read_latency, "ns")
		print("crossbar_read_energy:", self.xbar_read_energy, "nJ")
		print("crossbar_write_power:", self.xbar_write_power, "W")
		print("crossbar_write_latency:", self.xbar_write_latency, "ns")
		print("crossbar_write_energy:", self.xbar_write_energy, "nJ")

	
def xbar_test():
	print("load file:",test_SimConfig_path)
	_xbar = crossbar(test_SimConfig_path)
	print('------------')
	_xbar.xbar_read_config()
	_xbar.calculate_xbar_area()
	_xbar.calculate_xbar_read_latency()
	_xbar.calculate_xbar_read_power()
	_xbar.calculate_xbar_read_energy()
	_xbar.xbar_output()


if __name__ == '__main__':
	xbar_test()
