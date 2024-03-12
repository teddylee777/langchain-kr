import re
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def extract_abr(abr):
    youtube_audio_pattern = re.compile(r"\d+")
    kbps = youtube_audio_pattern.search(abr)
    if kbps:
        kbps = kbps.group()
        return int(kbps)
    else:
        return 0


def get_audio_filepath(filename):
    # audio 폴더가 없으면 생성
    if not os.path.isdir("audio"):
        os.mkdir("audio")

    # 현재 스크립트의 절대 경로 얻기
    current_directory = os.path.abspath("")

    # 파일 경로 생성
    audio_file_path = os.path.join(current_directory, "audio", filename)

    return audio_file_path


def convert_mp4_to_wav(mp4_file_path, wav_file_path):
    # MP4 파일 로드
    audio_clip = AudioFileClip(mp4_file_path)

    # WAV 형식으로 오디오 추출 및 저장
    audio_clip.write_audiofile(wav_file_path, fps=44100, nbytes=2, codec="pcm_s16le")


def download_audio_from_youtube(link):
    # YouTube 객체 생성
    yt = YouTube(link)

    # mp4 오디오만 필터링
    mp4_files = dict()

    # "audio/mp4" 타입의 스트림만 필터링
    for stream in yt.streams.filter(only_audio=True):
        mime_type = stream.mime_type
        abr = stream.abr
        if mime_type == "audio/mp4":
            abr = extract_abr(abr)
            mp4_files[abr] = stream

    # 키를 기준으로 정렬
    sorted_keys = sorted(mp4_files.keys())
    # 가장 큰 키를 사용하여 값 가져오기
    largest_value = mp4_files[sorted_keys[-1]]
    filename = largest_value.download()

    # 현재 스크립트의 절대 경로 얻기
    current_directory = os.path.abspath("")

    new_filename = os.path.basename(filename.replace(".mp4", ".wav"))

    new_filepath = os.path.join(current_directory, "audio", new_filename)

    # mp4 파일을 wav 파일로 변환
    convert_mp4_to_wav(filename, new_filepath)

    # mp4 파일 삭제
    os.remove(filename)
    return new_filepath


def extract_audio_from_video(video_filepath):
    # MP4 파일 로드
    video = VideoFileClip(video_filepath)
    audio_filepath = get_audio_filepath(video_filepath.replace(".mp4", ".wav"))
    video.audio.write_audiofile(audio_filepath)
    return audio_filepath


class AudioChunk:
    def __init__(self, filepath, min_silence_len=350, silence_thresh=-35):
        self.audio = AudioSegment.from_file(filepath, format="wav")
        self.filepath = filepath
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.detect_nonsilent_from_audio()

    @staticmethod
    def make_audio_chunks(audio, non_silent_times):
        audio_chunks = []
        for start, end in non_silent_times:
            audio_chunks.append((audio[start:end], start, end))
        return audio_chunks

    def detect_nonsilent_from_audio(self):
        non_silent_audio_times = detect_nonsilent(
            self.audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
        )

        non_silent_audios_output = AudioSegment.empty()

        for i in range(len(non_silent_audio_times)):
            non_silent_audios_output += self.audio[
                non_silent_audio_times[i][0] : non_silent_audio_times[i][1]
            ]
        self.audio_chunks = self.make_audio_chunks(self.audio, non_silent_audio_times)
        self.non_silent_audios_output = non_silent_audios_output
        print(f"분석에 사용할 전체 오디오 조각 개수: {len(non_silent_audio_times)}")

    def audio_splits(self, split_time=100):
        splits = int(self.audio.duration_seconds // split_time + 1)
        audios = []
        for s in range(splits):
            start = s * split_time * 1000
            end = start + split_time * 1000
            audios.append(self.audio[start:end])
        return audios
