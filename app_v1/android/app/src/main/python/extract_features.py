from librosa_wrapper import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms
import soundfile as sf
import numpy as np
import sys

DEFAULT_FRAME_SIZE = 512

def extract_features(file_name_and_tick):
    (file_name, tick) = file_name_and_tick.split("|")

    #with open(file_name, 'rb') as f:
    sr = 44100
    with sf.SoundFile(file_name, mode='rb', samplerate=44100, channels=1, subtype='PCM_16', format='RAW') as sf_desc:
        sr_native = sf_desc.samplerate
        sf_desc.seek(int(tick) * sr_native)
        frame_duration = sr_native * 5 # 5 seconds
        audio_data = sf_desc.read(frames=frame_duration, dtype=np.float32, always_2d=False).T
    
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

    np.savetxt(sys.stdout, mean_feat.ravel(), newline=' ')
    #return np.array_str(mean_feat.ravel())
