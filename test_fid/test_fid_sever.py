import time
from io import BytesIO
import pickle
import os
from shutil import copy
import socket

from test_fid.cal_fid_score_in_new_process import FidCalculator


class FidCalculatorSever:
    def __init__(self, real_img_dir, gpu_id, redirect_file='fid_records.txt', best_info_path=None, port=1234):
        host = "localhost"

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((host, port))
        self.fid_calculator = FidCalculator(
            real_img_dir,
            redirect_file,
            best_info=best_info_path,
            device=f'cuda:{gpu_id}'
        )

    def run(self, model_class, model_args, dataloader, datalength, model_path, best_path="", fn=lambda x: x):

        while True:
            print('Waiting for signal...')
            recv_data = self.udp_socket.recvfrom(1024)
            recv_data = recv_data[0]
            io_ = BytesIO(recv_data)
            params = pickle.load(io_)
            io_.close()

            if params.test_signal:
                for i in range(40):  # if contour reading problem, initialize the model 40 times at most.
                    try:
                        model = model_class(model_args)
                        break
                    except Exception as e:
                        time.sleep(1)
                        print(e)
                        print(f"Loading model error, reloading for time {i}")
                else:
                    raise FileNotFoundError("reading Model error")

                model.eval()
                is_best = self.fid_calculator.fid(model, datalength, dataloader, params.epoch, fn=fn)
                if is_best:
                    if best_path == "":
                        best_path = "best_net_G.pth"
                    dest_path = os.path.join(os.path.dirname(model_path), best_path)
                    copy(model_path, dest_path)
                del model
            else:
                break
