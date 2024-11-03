import os
import json
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip


#make sure folders exist
whistles_dir = "whistles"
marked_clips_dir = "marked_clips"


if not os.path.exists(marked_clips_dir):
    os.makedirs(marked_clips_dir)






#find start of 4th peak
def find_fourth_peak_start(audio_segment, threshold_db=-20):

    #get how loud each part is
    loudness = np.array([audio_segment[i:i + 1].dBFS for i in range(0, len(audio_segment))])



    #find the parts above threshold loudness
    peaks, _ = find_peaks(loudness, height=threshold_db, distance=500)

    #if 4 peaks find 4th
    if len(peaks) >= 4:
        return peaks[3], loudness, peaks
    
    return None, None, None








#go to ech folder video
for filename in os.listdir(whistles_dir):


    if filename.endswith(".mp4"):

        video_path = os.path.join(whistles_dir, filename)

        #get audio from the video
        video_clip = VideoFileClip(video_path)
        audio = video_clip.audio
        audio_path = os.path.join(whistles_dir, "temp_audio.wav")
        audio.write_audiofile(audio_path)
        audio_segment = AudioSegment.from_wav(audio_path)





        #call function for 4th peak
        fourth_peak_start_time, loudness, peaks = find_fourth_peak_start(audio_segment)

        if fourth_peak_start_time is not None:

            #sort folders for each video
            clip_dir = os.path.join(marked_clips_dir, filename[:-4])
            os.makedirs(clip_dir, exist_ok=True)

            #save aduio
            marked_audio_path = os.path.join(clip_dir, f"{filename[:-4]}.wav")
            audio_segment.export(marked_audio_path, format="wav")



            #put time stamp in json
            json_data = {"4th_high_start_time_ms": int(fourth_peak_start_time)}
            json_path = os.path.join(clip_dir, "whistle_data.json")

            #save
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)





            #plot decibelss and peaks
            time_axis = np.arange(len(loudness))

            plt.plot(time_axis, loudness, label="Loudness (dBFS)")
            plt.scatter(peaks, loudness[peaks], color='red', label="Detected Peaks")
            plt.axvline(x=fourth_peak_start_time, color='green', linestyle='--', label="4th High Start")
            plt.title(f"Loudness with Peaks for {filename}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Loudness (dBFS)")
            plt.legend()








            #save plot
            plot_path = os.path.join(clip_dir, "loudness_plot.png")
            plt.savefig(plot_path)
            plt.close()

        #delete old audio file
        os.remove(audio_path)







#close down
video_clip.close()

print("done")
