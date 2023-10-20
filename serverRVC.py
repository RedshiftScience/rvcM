hubert_model = None
tgt_sr = None
net_g = None
vc = None
cpt = None
from config import Config
import resampy

config = Config()
import os
# from memory_profiler import profile

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from vc_infer_pipeline import VC
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from fairseq import checkpoint_utils
from concurrent import futures
import numpy as np
import ASH_grpc_pb2
import ASH_grpc_pb2_grpc
import grpc
import time
import threading

processing_lock = threading.Lock()


class ServerRVCService(ASH_grpc_pb2_grpc.ServerTTSService):
    def ProcessAudio(self, request, context):
        # Get the lock'
        with processing_lock:
            
            audio_bytes = request.audio_bytes

        
            # Generate RVC config dictionary from request fields
            rvc_config = {
                "sid": request.sid,
                "audio_bytes": audio_bytes,
                "f0_up_key": request.f0_up_key,
                "f0_file": request.f0_file,
                "f0_method": request.f0_method,
                "file_index": request.file_index,
                "file_index2": request.file_index2,
                "index_rate": request.index_rate,
                "filter_radius": request.filter_radius,
                "resample_sr": request.resample_sr,
                "rms_mix_rate": request.rms_mix_rate,
                "protect": request.protect,
                "version": request.version,
                "tgt_sr": request.tgt_sr,
            }

            # Process the audio bytes using RVC logic
            start = time.time()
            processed_audio_bytes = gen_RVC(**rvc_config)
            print("time: ", time.time() - start)
            # Create and return the response
            response = ASH_grpc_pb2.ServerRVCResponse(audio_bytes=processed_audio_bytes)
            return response

# ===================== RVC =====================


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def init_models(model_name):
    global n_spk, tgt_sr, net_g, vc, cpt, version, config
    print("model_name: ", model_name)

    # configs
    configs = {
        "device": "cuda",  # cuda:0
        "is_half": False,
        "n_cpu": 8,
        "gpu_name": None,
        "gpu_mem": 6,  # 6
        "python_cmd": "python",
        "listen_port": 7865,
        "iscolab": False,
        "noparallel": False,
        "noautoopen": False,
        "x_pad": 3,
        "x_query": 10,
        "x_center": 60,
        "x_max": 65,
    }

    config.device = configs["device"]
    config.is_half = configs["is_half"]
    config.n_cpu = configs["n_cpu"]
    config.gpu_name = configs["gpu_name"]
    config.gpu_mem = configs["gpu_mem"]
    config.python_cmd = configs["python_cmd"]
    config.listen_port = configs["listen_port"]
    config.iscolab = configs["iscolab"]
    config.noparallel = configs["noparallel"]
    config.noautoopen = configs["noautoopen"]
    config.x_pad = configs["x_pad"]
    config.x_query = configs["x_query"]
    config.x_center = configs["x_center"]
    config.x_max = configs["x_max"]

    cpt = torch.load(model_name, map_location="cpu")

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def gen_RVC(
    sid,
    audio_bytes,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    version,
    tgt_sr,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global net_g, vc, hubert_model

    # print all the input values

    print("sid:", sid)
    print("f0_up_key:", f0_up_key)
    print("f0_file:", f0_file)
    print("f0_method:", f0_method)
    print("file_index:", file_index)
    print("file_index2:", file_index2)
    # print("file_big_npy:", file_big_npy)
    print("index_rate:", index_rate)
    print("filter_radius:", filter_radius)
    print("resample_sr:", resample_sr)
    print("rms_mix_rate:", rms_mix_rate)
    print("protect:", protect)
    print("version:", version)
    print("tgt_sr:", tgt_sr)
    # print("net_g:", net_g)
    # print("vc:", vc)
    # print("hubert_model:", hubert_model)

    input_audio_path = "ashera"

    f0_up_key = int(f0_up_key)

    audio_in = np.frombuffer(audio_bytes, dtype=np.float32)
    # reverse from (1,x) to (x,)\
    print("before audio_in.shape: ", audio_in.shape)
    # print dtype
    # print("before audio_in.shape: ", audio_in.dtype)

    # audio_in = audio_in[0]

    # print("after audio_in.shape: ", audio_in.shape)
    # resample 44100 -> 16000 using resampy
    audio = resampy.resample(audio_in, 22050, 16000)

    try:
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            128,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr

        print(audio_opt.shape)
        return audio_opt.tobytes()
    except Exception as e:
        raise e
        return None

# @profile
def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[
            (
                "grpc.max_receive_message_length",
                100 * 1024 * 1024,
            )  # Set the maximum receive message length to 100 MB
        ],
    )
    ASH_grpc_pb2_grpc.add_ServerRVCServiceServicer_to_server(ServerRVCService(), server)
    server.add_insecure_port(
        "[::]:50052"
    )  # Use a different port number for the RVC server
    server.start()
    server.wait_for_termination()



import requests

def get_model(name):
    model_name = name.split("/")[-1]
    # check if url or name
    if name.startswith("http"):
        # download the model
        print("Downloading model from url...")
        resp = requests.get(name)
        with open(model_name, "wb") as f:
            f.write(resp.content)
     
    elif name.startswith("./"):
        model_name = name.split("/")[-1]

    else:
        # download from base url 
        base = "https://files.redshiftscience.com/api/public/dl/lMWjjCRp/rvcM/"

        # download the model
        print("Downloading model from url...")
        resp = requests.get(base + name)
        with open(model_name, "wb") as f:
            f.write(resp.content)

    return model_name

if __name__ == "__main__":
    # get model name or url
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="ashera_97k.pth",
    #     help="Model name or url to download model from. ",
    # )
    # args = parser.parse_args()
    # # load model
    

    init_models(get_model(config.model_name))
    serve()
