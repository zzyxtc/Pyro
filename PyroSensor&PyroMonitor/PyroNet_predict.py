import socket
import struct
import os
import numpy as np
import torch
import joblib
import json
import time
import subprocess
import concurrent.futures
import sys
from sklearn.preprocessing import MinMaxScaler
from models import EnhancedCNNSeq2Seq

 
# Netlink配置
NETLINK_USER = 31
NLMSG_ALIGNTO = 4
NLMSG_HDRLEN = 16
INIT_MSG_TYPE = 0x10
AVG_TEMP_MSG_TYPE = 0x11 
TEMP_DATA_MSG_TYPE = 0x12 


# 模型配置参数
CONFIG = {
    "seq_length": 30,
    "pred_length": 20,
    "num_features": 21,
    "cnn_channels": 64,
    "heat_threshold": 90,
    "savgol_window": 5,
    "savgol_polyorder": 2,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3
}

if CONFIG["savgol_window"] % 2 == 0:
    CONFIG["savgol_window"] += 1

# 完整的温度数据结构格式字符串
TEMP_DATA_FORMAT = "=i i B I I B B B Q Q Q Q Q Q I I I I I I I I I"
TEMP_DATA_SIZE = struct.calcsize(TEMP_DATA_FORMAT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


class HybridTemperaturePredictor:
    def __init__(self):
        self.model_dtype = torch.half if (DEVICE.type == 'cuda') else torch.float32
        # 先加载内核模块
        self.load_kernel_module()
        # 然后初始化Netlink
        self.initialize_netlink()
        # 加载模型和预处理工具
        self.model, self.scalers, self.feature_columns = self.load_models()

        # 为每个核心维护数据缓冲区
        self.core_buffers = {}

        self.savgol_kernels = {}
        self.precompute_savgol_kernels()

        self.scaler_mins = np.zeros(len(self.feature_columns))
        self.scaler_scales = np.zeros(len(self.feature_columns))
        for j, feature_name in enumerate(self.feature_columns):
            if feature_name in self.scalers:
                self.scaler_mins[j] = self.scalers[feature_name].min_[0]
                self.scaler_scales[j] = self.scalers[feature_name].scale_[0]

        print("PyroNet initialized")
        print("Model running on:", DEVICE)
        print("Model dtype:", self.model_dtype)
        print("Savitzky-Golay configs: window size={}, polyorder={}".format(
            CONFIG['savgol_window'], CONFIG['savgol_polyorder']))


    def precompute_savgol_kernels(self):
        window_sizes = [CONFIG["savgol_window"]]
        polyorder = CONFIG["savgol_polyorder"]

        for window_size in window_sizes:
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < polyorder + 1:
                continue

            try:
                from scipy.signal import savgol_coeffs
                kernel = savgol_coeffs(window_size, polyorder, deriv=0)
            except ImportError:
                from scipy.linalg import lstsq
                x = np.arange(window_size)
                A = np.vander(x, polyorder + 1)
                b = np.zeros(window_size)
                b[(window_size - 1) // 2] = 1.0
                kernel, _, _, _ = lstsq(A, b)

            self.savgol_kernels[window_size] = kernel

    def load_models(self):

        model = EnhancedCNNSeq2Seq(
            input_size=CONFIG["num_features"],
            hidden_size=CONFIG["hidden_size"],
            cnn_channels=CONFIG["cnn_channels"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            pred_length=CONFIG["pred_length"]
        ).to(DEVICE)

        model.load_state_dict(torch.load('PyroNet_model/PyroNet.pth', map_location=DEVICE))

        if DEVICE.type == 'cuda':
            model = model.half()  # FP16量化

        model.eval()
        try:
            example_input = torch.randn(1, CONFIG["seq_length"], CONFIG["num_features"],
                                        dtype=self.model_dtype, device=DEVICE)
            model = torch.jit.trace(model, example_input)
            print("TorchScript optimization applied successfully")
        except Exception as e:
            print(f"TorchScript conversion failed: {e}. Using original PyTorch model.")

        with open('PyroNet_model/scalers.json', 'r') as f:
            scaler_data = json.load(f)
        scalers = {}
        for feature, params in scaler_data.items():
            scaler = MinMaxScaler()
            scaler.min_ = np.array(params['min_'])
            scaler.scale_ = np.array(params['scale_'])
            scaler.data_min_ = np.array(params['data_min_'])
            scaler.data_max_ = np.array(params['data_max_'])
            scaler.feature_range = tuple(params['feature_range'])
            scalers[feature] = scaler

        # 加载特征列顺序
        with open('PyroNet_model/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)

        return model, scalers, feature_columns


    def load_kernel_module(self):
        """插入内核模块并传递当前PID作为参数"""
        pid = os.getpid()
        module_path = "./Pyro.ko"

        print("Inserting kernel module with PID: {}".format(pid))
        try:
            result = subprocess.run(
                ["sudo", "insmod", module_path, "user_pid={}".format(pid)],
                capture_output=True,
                text=True,
                check=True
            )
            print("Kernel module inserted successfully")
        except subprocess.CalledProcessError as e:
            print("Failed to insert kernel module:")
            print("Error code:", e.returncode)
            print("Output:", e.stdout)
            print("Error:", e.stderr)
            sys.exit(1)

    def unload_kernel_module(self):
        print("Unloading kernel module...")
        try:
            subprocess.run(
                ["sudo", "rmmod", "Pyro"],
                capture_output=True,
                text=True,
                check=True
            )
            print("Kernel module unloaded successfully")
        except subprocess.CalledProcessError as e:
            print("Failed to unload kernel module:")
            print("Error code:", e.returncode)
            print("Output:", e.stdout)
            print("Error:", e.stderr)

    def initialize_netlink(self):
        self.sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_USER)

        # 增加接收缓冲区大小
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 10)  # 10MB
        # 增加发送缓冲区大小
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB

        self.sock.bind((os.getpid(), 0))

        # 发送初始化消息
        init_msg = self.build_nlmsg(INIT_MSG_TYPE, b'')
        self.sock.sendto(init_msg, (0, 0))
        print("Sent initialization message with PID: {}".format(os.getpid()))

    def nlmsg_align(self, len):
        return (len + NLMSG_ALIGNTO - 1) & ~(NLMSG_ALIGNTO - 1)

    def build_nlmsg(self, msg_type, payload):
        payload_len = len(payload)
        nlmsg_len = self.nlmsg_align(NLMSG_HDRLEN + payload_len)
        nlh = struct.pack("IHHII",
                          nlmsg_len,
                          msg_type,
                          0,
                          0,
                          os.getpid())
        padded_payload = payload.ljust(nlmsg_len - NLMSG_HDRLEN, b'\0')
        return nlh + padded_payload

    def parse_temperatures(self, data):
        nlh = data[:NLMSG_HDRLEN]
        hdr = struct.unpack("=IHHII", nlh)
        if hdr[1] != TEMP_DATA_MSG_TYPE:
            print("Received unknown message type: {}".format(hdr[1]))
            return None

        payload = data[NLMSG_HDRLEN:]
        msg_size = hdr[0] - NLMSG_HDRLEN
        num_samples = msg_size // TEMP_DATA_SIZE

        temps = []
        for i in range(num_samples):
            offset = i * TEMP_DATA_SIZE
            temp_data = payload[offset:offset + TEMP_DATA_SIZE]

            (logical_core, physical_core, coretemp, pkg_energy, dram_energy,
             cpu_freq, cpu_voltage, cpu_load, instruction_diff, cycle_diff,
             cache_diff, cache_miss_diff, bus_cycle_diff, branch_instruction_diff,
             branch_insMiss_diff, task_clock_diff, l1_data_read_diff,
             l1_data_write_diff, l1_data_miss_diff, l1_ins_miss_diff,
             l2_read_miss_diff, context_switches_diff, page_faults_diff) = struct.unpack(TEMP_DATA_FORMAT, temp_data)

            temps.append({
                'logical_core': logical_core,
                'physical_core': physical_core,
                'coretemp': coretemp,
                'pkg_energy': pkg_energy,
                'dram_energy': dram_energy,
                'cpu_freq': cpu_freq,
                'cpu_voltage': cpu_voltage,
                'cpu_load': cpu_load,
                'instruction_diff': instruction_diff,
                'cycle_diff': cycle_diff,
                'cache_diff': cache_diff,
                'cache_miss_diff': cache_miss_diff,
                'bus_cycle_diff': bus_cycle_diff,
                'branch_instruction_diff': branch_instruction_diff,
                'branch_insMiss_diff': branch_insMiss_diff,
                'task_clock_diff': task_clock_diff,
                'l1_data_read_diff': l1_data_read_diff,
                'l1_data_write_diff': l1_data_write_diff,
                'l1_data_miss_diff': l1_data_miss_diff,
                'l1_ins_miss_diff': l1_ins_miss_diff,
                'l2_read_miss_diff': l2_read_miss_diff,
                'context_switches_diff': context_switches_diff,
                'page_faults_diff': page_faults_diff
            })
        return temps



    def preprocess_data(self, core_id, core_data):
        feature_names = [
            'coretemp', 'pkg_energy', 'dram_energy', 'cpu_freq', 'cpu_voltage',
            'cpu_load', 'instruction_diff', 'cycle_diff', 'cache_diff',
            'cache_miss_diff', 'bus_cycle_diff', 'branch_instruction_diff',
            'branch_insMiss_diff', 'task_clock_diff', 'l1_data_read_diff',
            'l1_data_write_diff', 'l1_data_miss_diff', 'l1_ins_miss_diff',
            'l2_read_miss_diff', 'context_switches_diff', 'page_faults_diff'
        ]

        num_features = len(feature_names)
        data_array = np.zeros((len(core_data), num_features))

        for i, data_point in enumerate(core_data):
            for j, name in enumerate(feature_names):
                data_array[i, j] = data_point[name]

        window_size = CONFIG["savgol_window"]
        kernel = self.savgol_kernels.get(window_size)

        if kernel is not None:
            for j in range(num_features):
                feature_col = data_array[:, j]
                # 使用预计算的卷积核进行卷积
                data_array[:, j] = np.convolve(feature_col, kernel, mode='same')

        data_array -= self.scaler_mins
        data_array *= self.scaler_scales

        # 转换为张量
        tensor_data = torch.tensor(data_array, dtype=self.model_dtype, device=DEVICE).unsqueeze(0)
        return tensor_data

    def predict_temperature(self, core_id, core_data):
        if len(core_data) < CONFIG["seq_length"]:
            print("Core {}: Insufficient data ({} < {})".format(
                core_id, len(core_data), CONFIG['seq_length']))
            return None

        step_times = {}

        start_time = time.time()
        input_data = self.preprocess_data(core_id, core_data)
        step_times['preprocess'] = time.time() - start_time

        start_time = time.time()
        with torch.no_grad():
            pred = self.model(input_data)
            if isinstance(pred, tuple):
                pred = pred[0]
        step_times['inference'] = time.time() - start_time

        start_time = time.time()
        temp_scaler = self.scalers['Temperature (°C)']
        final_pred = temp_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        step_times['postprocess'] = time.time() - start_time

        return final_pred

    def handle_core_prediction(self, core_id):
        if len(self.core_buffers[core_id]) < CONFIG["seq_length"]:
            return

        prediction = self.predict_temperature(
            core_id,
            self.core_buffers[core_id][-CONFIG["seq_length"]:]  # 取最新数据
        )

        if prediction is not None:
            avg_pred_temp = np.mean(prediction)
            heat_risk = np.any(prediction > CONFIG["heat_threshold"])

            self.send_prediction(
                self.core_buffers[core_id][-1]['logical_core'],
                self.core_buffers[core_id][-1]['physical_core'],
                int(avg_pred_temp)
            )

    def send_prediction(self, logical_core, physical_core, avg_temp):
        payload = struct.pack("I I i", logical_core, physical_core, avg_temp)
        avg_msg = self.build_nlmsg(AVG_TEMP_MSG_TYPE, payload)
        self.sock.sendto(avg_msg, (0, 0))

    def run_mr(self):
        print("Starting hybrid temperature prediction multi threading service...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                while True:
                    data = self.sock.recv(1024 * 1024)
                    if not data:
                        time.sleep(0.1)
                        continue
                    temps = self.parse_temperatures(data)
                    if temps:
                        core_data_map = {}
                        for temp_data in temps:
                            core_id = temp_data['logical_core']
                            if core_id not in core_data_map:
                                core_data_map[core_id] = []
                            core_data_map[core_id].append(temp_data)

                        # 并行处理每个核心的预测
                        futures = []
                        for core_id, data_list in core_data_map.items():
                            if core_id not in self.core_buffers:
                                self.core_buffers[core_id] = []
                            self.core_buffers[core_id].extend(data_list)

                            if len(self.core_buffers[core_id]) > CONFIG["seq_length"]:
                                self.core_buffers[core_id] = self.core_buffers[core_id][-CONFIG["seq_length"]:]

                            # 仅在缓冲区满时预测
                            if len(self.core_buffers[core_id]) >= CONFIG["seq_length"]:
                                futures.append(executor.submit(self.handle_core_prediction, core_id))

                        for future in concurrent.futures.as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                print(f"Error in task execution for core {core_id}: {e}")
                    else:
                        print("No valid temperature data received.")
        except KeyboardInterrupt:
            print("\nTerminating...")
        except Exception as e:
            print(f"Error in prediction service: {e}")
        finally:
            print("Cleaning up resources...")
            self.sock.close()
            self.unload_kernel_module()
            print("Service terminated.")


if __name__ == "__main__":
    predictor = HybridTemperaturePredictor()
    predictor.run_mr()
