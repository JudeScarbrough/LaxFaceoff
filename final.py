import os
import json
import time
import pygame
import random
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import winsound
from collections import deque










#start pygame
pygame.init()
pygame.mixer.init()

#set window
window_width, window_height = 1400, 900
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Reaction Time Test")

#start opencv
cap = cv2.VideoCapture(0)

#allowance for moevement
MOVEMENT_THRESHOLD = 3
RUNNING_AVERAGE_WINDOW = 5
STILLNESS_TIME_MS = 1000
#directory
marked_clips_dir = "marked_clips"


#vars
reaction_times = []
movement_history = deque(maxlen=RUNNING_AVERAGE_WINDOW)
ball_moving = False
fault_occurred = False
previous_avg_point = None
user_still = True
graph_shown = False
num_reps = 10  #default reps
rep_counter = 1
test_started = False
graph_image_path = "reaction_times_graph.png"
start_over_button_shown = False












#function to detect opencv movement
def find_custom_color_pixels(frame, lower_bound, upper_bound):
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def is_ball_moving(distance):

    movement_history.append(distance)
    avg_movement = np.mean(movement_history)


    return avg_movement > MOVEMENT_THRESHOLD

def draw_average_circle(mask, previous_avg_point):
    white_pixels = cv2.findNonZero(mask)


    if white_pixels is not None:

        average_point = np.mean(white_pixels, axis=0)
        average_point = tuple(average_point[0].astype(int))



        distance = 0 if previous_avg_point is None else math.sqrt(
            (average_point[0] - previous_avg_point[0]) ** 2 +
            (average_point[1] - previous_avg_point[1]) ** 2
        )
        ball_moving = is_ball_moving(distance)
        return average_point, ball_moving
    return previous_avg_point, False








#function to play audio and measure times
def play_audio_and_monitor_movement(audio_path, json_path):
    global previous_avg_point, ball_moving, fault_occurred, rep_counter, user_still


    #load timestamp
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        last_elevated_time = data.get("4th_high_start_time_ms", None)



    if last_elevated_time is not None:
        try:

            sound = pygame.mixer.Sound(audio_path)
            audio_duration = sound.get_length() * 1000


        except Exception as e:

            print(f"Error loading audio: {e}")
            return



        #play
        sound.play()
        start_time = pygame.time.get_ticks()
        fault_occurred = False
        staying_still = True
        user_still = True
        reaction_recorded = False





        while pygame.mixer.get_busy():
            current_time = pygame.time.get_ticks() - start_time

            #check if within 1 sec before whistle
            if last_elevated_time - STILLNESS_TIME_MS <= current_time < last_elevated_time:
                #user has to be still
                if not user_still:

                    fault_occurred = True


                    turn_screen_red()
                    break



            #allow movement after the whistle and measure reaction time
            if current_time >= last_elevated_time:

                if not reaction_recorded and ball_moving:

                    reaction_time = current_time - last_elevated_time
                    reaction_times.append(reaction_time)

                    reaction_recorded = True





            #check for movement using OpenCV
            ret, frame = cap.read()

            if not ret:
                break

            mask = find_custom_color_pixels(frame, (155, 155, 155), (255, 255, 255))#pre tuned color mask
            average_point, ball_moving = draw_average_circle(mask, previous_avg_point)
            previous_avg_point = average_point





            #if movement is detected during stillness window
            if ball_moving and last_elevated_time - STILLNESS_TIME_MS <= current_time < last_elevated_time:
                user_still = False

            #pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()




            #update display camera feed mask view rep count
            screen.fill((0, 0, 0))  #clear screen

            #top left
            frame_resized = cv2.resize(frame, (window_width // 2, window_height // 2))
            raw_feed_surface = pygame.surfarray.make_surface(cv2.transpose(frame_resized))
            screen.blit(raw_feed_surface, (0, 0))




            #bottomleft
            mask_resized = cv2.resize(mask, (window_width // 2, window_height // 2))
            mask_surface = pygame.surfarray.make_surface(cv2.transpose(mask_resized))
            screen.blit(mask_surface, (0, window_height // 2))



            #right side
            font = pygame.font.Font(None, 50)
            rep_text = font.render(f"Rep: {rep_counter}/{num_reps}", True, (255, 255, 255))
            screen.blit(rep_text, (window_width // 2 + 50, 50))



            #display current reaction times
            for i, rt in enumerate(reaction_times):
                reaction_text = font.render(f"Reaction {i+1}: {rt} ms", True, (255, 255, 255))
                screen.blit(reaction_text, (window_width // 2 + 50, 120 + i * 40))

            pygame.display.update()

        if not fault_occurred:
            rep_counter += 1#increment rep if no fault









#function to turn the screen red temporarily during fault
def turn_screen_red():

    screen.fill((255, 0, 0))

    pygame.display.update()

    winsound.Beep(400, 1000)#play fault 400Hz for 1s

    pygame.time.delay(1000)#delay for the duration

#function to create save reaction time bar graph
def create_and_save_reaction_time_graph():
    if not reaction_times:
        return



    #matplotlib
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    #plot bar chart
    ax.bar(range(1, len(reaction_times) + 1), reaction_times, color='blue')


    #plot average line
    avg_time = np.mean(reaction_times)
    ax.axhline(y=avg_time, color='red', linestyle='--', label=f'Average: {avg_time:.2f} ms')





    #set labels
    ax.set_xlabel("Reps")
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_title("Reaction Time Per Rep")
    ax.legend()

    #add reaction time below each bar
    for i, v in enumerate(reaction_times):
        ax.text(i + 1, v + 5, f'{v:.2f} ms', ha='center')


    #save graph
    plt.savefig(graph_image_path)
    plt.close()







#display reaction time graph
def display_reaction_time_graph():

    global start_over_button_shown
    if not os.path.exists(graph_image_path):

        create_and_save_reaction_time_graph()

    #load image
    graph_image = pygame.image.load(graph_image_path)

    graph_image = pygame.transform.scale(graph_image, (window_width, window_height))
    screen.blit(graph_image, (0, 0))

    #start over button only if not already displayed

    font = pygame.font.Font(None, 60)
    button_text = font.render("Start Over", True, (255, 255, 255))

    pygame.draw.rect(screen, (0, 128, 255), [window_width // 2 - 150, window_height - 150, 300, 100])
    screen.blit(button_text, (window_width // 2 - 80, window_height - 120))

    start_over_button_shown = True




    pygame.display.update()








#reset everything and start over
def start_over():
    global rep_counter, reaction_times, fault_occurred, previous_avg_point, user_still, graph_shown, test_started, start_over_button_shown


    rep_counter = 1
    reaction_times = []


    fault_occurred = False
    previous_avg_point = None
    user_still = True
    graph_shown = False
    test_started = False
    start_over_button_shown = False







#display start screen with reps and start button
def start_screen():
    global num_reps, test_started



    while not test_started:
        screen.fill((0, 0, 0))

        #instructions and rep incrementer
        #change
        font = pygame.font.Font(None, 50)

        instructions = font.render("Adjust Reps (3-50) and Click Start", True, (255, 255, 255))
        screen.blit(instructions, (window_width // 2 - 300, 100))


        rep_text = font.render(f"Number of Reps: {num_reps}", True, (255, 255, 255))
        screen.blit(rep_text, (window_width // 2 - 100, 250))


        pygame.draw.rect(screen, (0, 128, 255), [window_width // 2 - 150, 450, 300, 100])
        start_text = font.render("Start", True, (255, 255, 255))
        screen.blit(start_text, (window_width // 2 - 40, 480))


        pygame.draw.rect(screen, (255, 255, 255), [window_width // 2 + 50, 350, 50, 50])#up
        pygame.draw.rect(screen, (255, 255, 255), [window_width // 2 - 100, 350, 50, 50])#down
        plus_text = font.render("+", True, (0, 0, 0))
        minus_text = font.render("-", True, (0, 0, 0))
        screen.blit(plus_text, (window_width // 2 + 65, 360))
        screen.blit(minus_text, (window_width // 2 - 85, 360))










        pygame.display.update()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                exit()



            elif event.type == pygame.MOUSEBUTTONDOWN:

                #start clicked
                if window_width // 2 - 150 <= pygame.mouse.get_pos()[0] <= window_width // 2 + 150 and 450 <= pygame.mouse.get_pos()[1] <= 550:
                    test_started = True


                #up or down clicked
                elif window_width // 2 + 50 <= pygame.mouse.get_pos()[0] <= window_width // 2 + 100 and 350 <= pygame.mouse.get_pos()[1] <= 400 and num_reps < 50:
                    num_reps += 1
                elif window_width // 2 - 100 <= pygame.mouse.get_pos()[0] <= window_width // 2 - 50 and 350 <= pygame.mouse.get_pos()[1] <= 400 and num_reps > 3:
                    num_reps -= 1











#main/start
def main():
    global fault_occurred
    audio_files = []



    
    #collect clips from directory
    for folder in os.listdir(marked_clips_dir):
        folder_path = os.path.join(marked_clips_dir, folder)
        if os.path.isdir(folder_path):
            audio_path = None
            json_path = None
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    audio_path = os.path.join(folder_path, file)
                if file.endswith(".json"):
                    json_path = os.path.join(folder_path, file)
            if audio_path and json_path:
                audio_files.append((audio_path, json_path))

    start_screen()

    #randomize audio clips
    random.shuffle(audio_files)
    audio_files = audio_files[:num_reps]

    
    valid_reps = 0
    while valid_reps < num_reps:
        audio_path, json_path = audio_files[valid_reps % len(audio_files)]
        play_audio_and_monitor_movement(audio_path, json_path)

        if not fault_occurred:
            valid_reps += 1




    create_and_save_reaction_time_graph()
    display_reaction_time_graph()


    while True:
        display_reaction_time_graph()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()




            elif event.type == pygame.MOUSEBUTTONDOWN:





                if window_width // 2 - 150 <= pygame.mouse.get_pos()[0] <= window_width // 2 + 150 and window_height - 150 <= pygame.mouse.get_pos()[1] <= window_height - 50:
                    start_over()
                    main()













#only run if running on same file
if __name__ == '__main__':
    main()


cap.release()
cv2.destroyAllWindows()
pygame.quit()
