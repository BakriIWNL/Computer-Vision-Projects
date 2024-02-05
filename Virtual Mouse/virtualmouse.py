import cv2
import mediapipe as mp
import math
import screeninfo
import pyautogui

# Initialize Mediapipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open video capture
cap = cv2.VideoCapture(0)
# Get the primary screen
screen = screeninfo.get_monitors()[0]


# Get the screen resolution
screen_width = screen.width
screen_height = screen.height
flag = 0
clicks = 0
unclicks = 0
rect_width = 400
rect_height = 200
rect_x = 100
rect_y = 200
map_x = screen_width / rect_width
map_y = screen_height / rect_height
print(rect_x,rect_x)
print(f"Screen resolution: {screen_width}x{screen_height}")


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read video")
            break

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), 2)
        # Process the image with Mediapipe
        results = hands.process(image_rgb)
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw red dots on the fingertips
                for finger_tip in [4, 8, 12, 16, 20]:
                    x = int(hand_landmarks.landmark[finger_tip].x * image.shape[1])
                    y = int(hand_landmarks.landmark[finger_tip].y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                
                # Get the coordinates of thumb and index finger
                thumb_x = int(hand_landmarks.landmark[4].x * image.shape[1])
                thumb_y = int(hand_landmarks.landmark[4].y * image.shape[0])
                index_x = int(hand_landmarks.landmark[8].x * image.shape[1])
                index_y = int(hand_landmarks.landmark[8].y * image.shape[0])
                middle_x = int(hand_landmarks.landmark[12].x * image.shape[1])
                middle_y = int(hand_landmarks.landmark[12].y * image.shape[0])

                
                # Calculate the distance between thumb and index finger
                distance = math.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)
                # Draw a line between thumb and index finger
                cv2.line(image, (thumb_x, thumb_y), (middle_x, middle_y), (0, 0, 255), 2)
                if(flag == 1):
                    unclicks += 1
                    # print(unclicks)
                # Check if the distance is greater than 5cm
                if distance < 20:
                    cv2.line(image, (thumb_x, thumb_y), (middle_x, middle_y), (0, 255, 0), 2)
                    flag = 1
                    if(unclicks > 2 and unclicks < 6):
                        flag = 0
                        pyautogui.click()
                        pyautogui.click()
                        cv2.putText(image, "Double click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        # print("Double Click")
                    unclicks = 0                
                if(unclicks > 6):
                    flag = 0
                    pyautogui.click()
                    # print("Click")
                    cv2.putText(image, "Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    unclicks = 0
                    
                # Move mouse cursor to (500, 500)
                if(index_x > rect_x and index_x < rect_x + rect_width and index_y > rect_y and index_y < rect_y + rect_height):
                    pyautogui.moveTo((index_x - rect_x) * map_x, (index_y - rect_y) * map_y)
                    # print((index_x - rect_x) * map_x, (index_y - rect_y) * map_y)

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()        
