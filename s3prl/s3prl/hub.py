from s3prl.upstream.apc.hubconf import *
from s3prl.upstream.audio_albert.hubconf import *
from s3prl.upstream.baseline.hubconf import *
# from s3prl.upstream.byol_a.hubconf import *  # FIXME: (Leo)
from s3prl.upstream.cpc.hubconf import *
from s3prl.upstream.data2vec.hubconf import *
from s3prl.upstream.decoar.hubconf import *
from s3prl.upstream.decoar_layers.hubconf import *
from s3prl.upstream.decoar2.hubconf import *
from s3prl.upstream.distiller.hubconf import *
from s3prl.upstream.example.hubconf import *
from s3prl.upstream.hubert.hubconf import *
# from s3prl.upstream.lighthubert.hubconf import *  # FIXME: (Leo)
#from s3prl.upstream.log_stft.hubconf import *
from s3prl.upstream.mockingjay.hubconf import *
from s3prl.upstream.mos_prediction.hubconf import *
from s3prl.upstream.npc.hubconf import *
from s3prl.upstream.pase.hubconf import *
from s3prl.upstream.roberta.hubconf import *
from s3prl.upstream.tera.hubconf import *
from s3prl.upstream.unispeech_sat.hubconf import *
from s3prl.upstream.vq_apc.hubconf import *
from s3prl.upstream.vq_wav2vec.hubconf import *
from s3prl.upstream.wav2vec.hubconf import *
from s3prl.upstream.wav2vec2.hubconf import *
from s3prl.upstream.wavlm.hubconf import *

from s3prl.downstream.timit_phone.hubconf import timit_posteriorgram


def options(only_registered_ckpt: bool = False):
    all_options = []
    for name, value in globals().items():
        torch_hubconf_policy = not name.startswith("_") and callable(value)
        if torch_hubconf_policy and name != "options":
            if only_registered_ckpt and (
                name.endswith("_local")
                or name.endswith("_url")
                or name.endswith("_gdriveid")
                or name.endswith("_custom")
            ):
                continue
            all_options.append(name)

    return all_options
