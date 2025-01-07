import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
from time import time
import os

class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_spectrogram_and_label(self, audio, label):
        spectrogram = self.get_spectrogram(audio)

        return spectrogram, label


class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label

class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        dbFSthres, 
        duration_thres
    ):
        self.sampling_rate = sampling_rate
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_length_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1

audio_buffer_old = None
audio_buffer_new = None
first_half_second = True
vad = VAD(16000, 0.032, 5, 300, 5500, -45, 0.2)

def transform(indata):
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    tf_indata = tf.squeeze(tf_indata)
    tf_indata = tf_indata / tf.int16.max

    return tf_indata
        
def callback(indata, frames, callback_time, status):
    global first_half_second
    global audio_buffer_old
    global audio_buffer_new
    global vad

    if first_half_second == True:
        audio_buffer_new = transform(indata)
        first_half_second = False
        return
    
    audio_buffer_old = audio_buffer_new
    audio_buffer_new = transform(indata)

    audio_buffer = tf.concat([audio_buffer_old, audio_buffer_new], axis = 0)

    print(audio_buffer)

    store_audio = vad.is_silence(audio_buffer)


    if store_audio == 0:
        timestamp = time()
        print(f'{timestamp} - NOT SILENCE')
        write(f'{timestamp}.wav', 16000, (audio_buffer * tf.int16.max).numpy())
        filesize_in_bytes = os.path.getsize(f'{timestamp}.wav')
        filesize_in_kb = filesize_in_bytes / 1024
        print(f'Size: {filesize_in_kb} KB')
    else:
        timestamp = time()
        print(f'{timestamp} - SILENCE')



if __name__ == '__main__':

    with sd.InputStream(
    device = 0,
    channels = 1, 
    dtype = 'int16', 
    samplerate = 16000, 
    callback=callback, 
    blocksize=8000):

        while True :
            key = input('Enter: ')
            if key in ('q', 'Q'):
                print('Stop recording.')
                break

            if key in ('p', 'P'):
                store_audio = False