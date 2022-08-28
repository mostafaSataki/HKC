from .FileUtility import *
from tqdm import tqdm
import os
from pydub import AudioSegment
import webrtcvad
import collections
import contextlib
import wave
import tempfile
import  librosa
import soundfile as sf
from HKC.Utility import *
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Manager
import random
class AudioFrame(object):
  def __init__(self, bytes, timestamp, duration):
    self.bytes = bytes
    self.timestamp = timestamp
    self.duration = duration

class AudioUtility:
  @staticmethod
  def stereo2Mono(src_filename, dst_filename,audio_format ="wav"):
    sound = AudioSegment.from_wav(src_filename)
    sound = sound.set_channels(1)
    sound.export(dst_filename, format=audio_format)

  @staticmethod
  def changeAudioProperties(src_filename, dst_filename,audio_format ="wav",num_channels = None,sample_rate = None,sample_width = None):
    if FileUtility.checkIsVideo(filename):
      sound = AudioSegment.from_file(filename)
    elif FileUtility.checkIsAudio(filename):
      sound = AudioSegment.from_file(filename)
    else:
      return None, None

    if num_channels and sound.channels != num_channels:
      sound.set_channels(num_channels)
    if sample_rate and sound.frame_rate != sample_rate:
      sound.set_frame_rate(sample_rate)
    if sample_width and sound.sample_width != sample_width:
      sound.set_sample_width(sample_width)


    sound.export(dst_filename, format=audio_format)

  @staticmethod
  def stereo2MonoBatch(src_filenames, dst_filenames):
    for i in tqdm(range(len(src_filenames)), ncols=100):
      src_filename = src_filenames[i]
      dst_filename = dst_filenames[i]
      stereo2Mono(src_filename, dst_filename)

  @staticmethod
  def changeAudioPropertiesBatch(src_filenames, dst_filenames,audio_format ="wav",num_channels = None,sample_rate = None,sample_width = None):
    for i in tqdm(range(len(src_filenames)), ncols=100):
      src_filename = src_filenames[i]
      dst_filename = dst_filenames[i]
      changeAudioProperties(src_filename, dst_filename,audio_format , num_channels , sample_rate , sample_width )

  @staticmethod
  def stereo2MonoFolder(src_path, dst_path):
    src_files = FileUtility.getFolderAudioFiles(src_path)
    dst_files = FileUtility.getDstFilenames(src_files, dst_path)
    stereo2MonoBatch(src_files, dst_files)

  @staticmethod
  def changeAudioPropertiesFolder(src_path, dst_path,audio_format ="wav",num_channels = None,sample_rate = None,sample_width = None):
    src_files = FileUtility.getFolderAudioFiles(src_path)
    dst_files = FileUtility.getDstFilenames(src_files, dst_path)

    stereo2MonoBatch(src_files, dst_files,audio_format , num_channels , sample_rate , sample_width )

  @staticmethod
  def readWave(filename):
      with contextlib.closing(wave.open(filename, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

  @staticmethod
  def readWave2(filename,num_channels = None,sample_rate = None,sample_width = None):
    if FileUtility.checkIsVideo(filename) or FileUtility.checkIsAudio(filename):
        sound = AudioSegment.from_file(filename)
        change = False
        if num_channels and sound.channels != num_channels:
          sound = sound.set_channels(num_channels)
          change = True
        if sample_rate and sound.frame_rate != sample_rate:
          sound = sound.set_frame_rate(sample_rate)
          change = True
        if sample_width and sound.sample_width != sample_width:
          sound = sound.set_sample_width(sample_width)
          change = True

        if change :
          temp_name = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".wav")
          sound.export(temp_name, "wav")
          sound = AudioSegment.from_file(temp_name)
          os.remove(temp_name)

    else :
      return None,None

    return sound.raw_data,sound.frame_rate



  @staticmethod
  def writeWave(filename, audio, sample_rate):
    with contextlib.closing(wave.open(filename, 'wb')) as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(sample_rate)
      wf.writeframes(audio)




  @staticmethod
  def frameGenerator(frame_duration_ms, audio, sample_rate):

    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
      yield AudioFrame(audio[offset:offset + n], timestamp, duration)
      timestamp += duration
      offset += n

  @staticmethod
  def vadCollector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    for frame in frames:
      is_speech = vad.is_speech(frame.bytes, sample_rate)

      sys.stdout.write('1' if is_speech else '0')
      if not triggered:
        ring_buffer.append((frame, is_speech))
        num_voiced = len([f for f, speech in ring_buffer if speech])
        if num_voiced > 0.9 * ring_buffer.maxlen:
          triggered = True
          sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
          for f, s in ring_buffer:
            voiced_frames.append(f)
          ring_buffer.clear()
      else:
        voiced_frames.append(frame)
        ring_buffer.append((frame, is_speech))
        num_unvoiced = len([f for f, speech in ring_buffer if not speech])
        if num_unvoiced > 0.9 * ring_buffer.maxlen:
          sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
          triggered = False
          yield b''.join([f.bytes for f in voiced_frames])
          ring_buffer.clear()
          voiced_frames = []
    if triggered:
      sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
      yield b''.join([f.bytes for f in voiced_frames])

  @staticmethod
  def segmentBySilence(src_filename,dst_path,mode = 3):
    # audio, sample_rate = AudioUtility.readWave(src_filename)
    audio, sample_rate = AudioUtility.readWave2(src_filename,num_channels= 1,sample_width= 2)

    vad = webrtcvad.Vad(mode)
    frames = AudioUtility.frameGenerator(30, audio, sample_rate)
    frames = list(frames)
    segments = AudioUtility.vadCollector(sample_rate, 30, 300, vad, frames)
    counter = 1
    tokens = FileUtility.getFileTokens(src_filename)
    for segment in segments:
      dst_filename = os.path.join(dst_path, tokens[1] + '_' + str(counter) + '.wav')
      AudioUtility.writeWave(dst_filename, segment, sample_rate)
      counter += 1


  @staticmethod
  def segmentBySilenceBatch(src_path,dst_path,mode = 3):
    src_filenames = FileUtility.getFolderAudioFiles(src_path)


    for i in tqdm(range(len(src_filenames)), ncols=100):
      src_filename = src_filenames[i]
      AudioUtility.segmentBySilence(src_filename,dst_path,3)

  @staticmethod
  def loadAudioFiles(filesname,sample_rate = 16000):
      result = []
      for i in tqdm(range(len(filesname)), ncols=100):
        filename = filesname[i]
        # sample,_ =  AudioUtility.readWave2(filename,num_channels=1,sample_rate=sample_rate)
        sample, _ = librosa.load(filename,mono=True,sr=sample_rate)
        result.append(sample)
      return  np.array(result)

  @staticmethod
  def saveAudioFiles(samples,filesname,sample_rate= 16000):
    for i in tqdm(range(len(filesname)), ncols=100):
        filename = filesname[i]
        sample = samples[i]

        sf.write(filename, sample, sample_rate)

        # AudioUtility.writeWave(filename,sample,sample_rate)

  @staticmethod
  def resample_files(src_files, dst_files, sample_rate=16000,progress = False):
      if progress:
         for i in tqdm(range(len(src_files)), ncols=100):
           src_file = src_files[i]
           dst_file = dst_files[i]
           data, sr = librosa.load(src_file, sr=None)
           data2 = librosa.resample(data,sr,sample_rate)

           sf.write(dst_file, data2, sample_rate)
      else :
        for i in range(len(src_files)):
          src_file = src_files[i]
          dst_file = dst_files[i]
          data, sr = librosa.load(src_file, sr=None)
          data2 = librosa.resample(data,sr,sample_rate)

          sf.write(dst_file, data2, sample_rate)

  @staticmethod
  def resample_files_mt(src_path, dst_path, sample_rate=16000,thread_count=4):
    FileUtility.copyFullSubFolders(src_path,dst_path)


    src_files = FileUtility.getFolderAudioFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files, src_path, dst_path)

    src_files_list =  Utility.breakList(src_files,thread_count)
    dst_files_list = Utility.breakList(dst_files, thread_count)

    threads = []
    for i in range(thread_count):
       threads.append( threading.Thread(target=AudioUtility.resample_files, args=(src_files_list[i],dst_files_list[i],sample_rate,i==0)))
       threads[i].start()

    for i in range(thread_count):
      threads[i].join()

  @staticmethod
  def extract_audio_db_info(src_files,res, progress = False):
    lenS = []
    durS = []
    srS = []
    if progress :
      for i in tqdm(range(len(src_files)), ncols=100):
        sample_file = src_files[i]

        data, sr = librosa.load(sample_file, sr=None)
        srS.append(sr)
        lenS.append(len(data))
        durS.append(librosa.get_duration(y=data, sr=sr))

    else :
      for i in range(len(src_files)):
        sample_file = src_files[i]

        data, sr = librosa.load(sample_file,sr=None)
        srS.append(sr)
        lenS.append(len(data))
        durS.append( librosa.get_duration(y=data, sr=sr))

    res['sr'] = srS
    res['dur'] = durS
    res['len'] = lenS


  @staticmethod
  def extract_audio_db_info_mt(src_path,thread_count = 4):
      sample_files = FileUtility.getFolderAudioFiles(src_path)

      sample_files_list = Utility.breakList(sample_files, thread_count)

      with Manager() as manager:
        all_res = []
        all_procs = []
        for i in range(thread_count):
           all_res.append( manager.dict())
           all_procs.append( Process(target=AudioUtility.extract_audio_db_info, args=(sample_files_list[i], all_res[i],i==0)))
           all_procs[i].start()

        for i in range(thread_count):
          all_procs[i].join()

        srS = all_res[0]['sr']
        lenS = all_res[0]['len']
        durS = all_res[0]['dur']

        for i in range(1,thread_count):
          srS = srS + all_res[i]['sr']
          lenS = lenS +  all_res[i]['len']
          durS = durS + all_res[i]['dur']

        srS = np.array(srS)
        lenS = np.array(lenS)
        durS = np.array(durS)

        min_sr = np.min(srS)
        max_sr = np.max(srS)
        min_len = np.min(lenS)
        max_len = np.max(lenS)
        min_dur = np.min(durS)
        max_dur = np.max(durS)

        return min_sr,max_sr,min_len,max_len,min_dur,max_dur

  @staticmethod
  def crop_audio(src_files,dst_files, begin, end,progress = False):
    if progress:
      for i in tqdm(range(len(src_files)), ncols=100):
        src_file = src_files[i]
        dst_file = dst_files[i]
        data,sr = librosa.load(src_file,sr=None)
        data = data[begin:end]
        sf.write(dst_file,data,sr)
    else :
      for i in range(len(src_files)):
        src_file = src_files[i]
        dst_file = dst_files[i]
        data,sr = librosa.load(src_file,sr=None)
        data = data[begin:end]
        sf.write(dst_file,data,sr)


  @staticmethod
  def crop_audio_mt(src_path,dst_path,begin,end,thread_count = 4):
    FileUtility.copyFullSubFolders(src_path,dst_path)
    src_files = FileUtility.getFolderAudioFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files, src_path, dst_path)

    src_files_list = Utility.breakList(src_files, thread_count)
    dst_files_list = Utility.breakList(dst_files, thread_count)

    threads = []
    for i in range(thread_count):
      threads.append(threading.Thread(target=AudioUtility.crop_audio,
                                      args=(src_files_list[i], dst_files_list[i], begin,end, i == 0)))
      threads[i].start()

    for i in range(thread_count):
      threads[i].join()







