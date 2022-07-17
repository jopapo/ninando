from io import SEEK_END
from librosa_wrapper import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms
import soundfile as sf
import numpy as np
import time

DEFAULT_FRAME_SIZE = 512

def monitor_file(file_name):
    tick = 1
    while True:
        start = time.time()

        try:
            mean_feat = extract_features(file_name)
            np.savetxt(f'{file_name}_{tick}.tick', mean_feat.ravel(), newline=' ')
        except FileNotFoundError:
            break
        except IndexError:
            pass
        
        tick = tick + 1

        # sleep only whats left for a full second
        sleep_seconds = 1 - (time.time() - start)
        time.sleep(sleep_seconds if sleep_seconds > 0 else 0)

def get_audio_data(file_name):
    sr = 44100
    try:
        with sf.SoundFile(file_name, samplerate=sr, channels=1, subtype='PCM_16', format='RAW') as sf_desc:
            frame_duration = sf_desc.samplerate * 5 # 5 seconds
            sf_desc.seek(-frame_duration, whence=SEEK_END)
            audio_data = sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T
            return audio_data, sr
    except RuntimeError as error:
        str_error = str(error)
        if str_error.startswith('Error opening'):
            raise FileNotFoundError("File not found") from error
        if str_error.startswith('Internal psf_fseek() failed'):
            raise IndexError("Index cannot go beyond the start") from error
        raise

def extract_features(file_name):
    audio_data, sr = get_audio_data(file_name)
    
    zcr_feat = zero_crossing_rate(y=audio_data, hop_length=DEFAULT_FRAME_SIZE)
    rmse_feat = rms(y=audio_data, hop_length=DEFAULT_FRAME_SIZE)
    mfcc_feat = mfcc(y=audio_data, sr=sr, n_mfcc=13)
    spectral_centroid_feat = spectral_centroid(y=audio_data, sr=sr, hop_length=DEFAULT_FRAME_SIZE)
    spectral_rolloff_feat = spectral_rolloff(y=audio_data, sr=sr, hop_length=DEFAULT_FRAME_SIZE, roll_percent=0.90)
    spectral_bandwidth_feat = spectral_bandwidth(y=audio_data, sr=sr, hop_length=DEFAULT_FRAME_SIZE)

    concat_feat = np.concatenate((zcr_feat,
                                    rmse_feat,
                                    mfcc_feat,
                                    spectral_centroid_feat,
                                    spectral_rolloff_feat,
                                    spectral_bandwidth_feat
                                    ), axis=0)

    mean_feat = np.mean(concat_feat, axis=1, keepdims=True).transpose()

    return mean_feat
