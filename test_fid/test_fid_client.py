import subprocess
import socket
from io import BytesIO
import pickle
from test_fid.constants import Param, BASE_DIR
import time



class FidCalculatorClient:
    def __init__(self, port=1234, ini_command="", **kwargs):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest_host = "localhost"
        self.dest_port = port
        cmd = ini_command + f" python {BASE_DIR}/fid_cal_server.py "
        for k, v in kwargs.items():
            cmd = cmd + f"--{k} " + f"{v} "
        self.process = subprocess.Popen(cmd, shell=True)

    def cal_fid(self, epoch):
        params = Param(test_signal=True, epoch=epoch)
        sen_data = BytesIO(pickle.dumps(params))
        self.udp_socket.sendto(sen_data.read(), (self.dest_host, self.dest_port))
        sen_data.close()

    def close(self):
        params = Param(test_signal=False, epoch=-1)
        sen_data = BytesIO(pickle.dumps(params))
        while self.process.poll() != 0:
            self.udp_socket.sendto(sen_data.read(), (self.dest_host, self.dest_port))
            time.sleep(3)
        self.udp_socket.close()

