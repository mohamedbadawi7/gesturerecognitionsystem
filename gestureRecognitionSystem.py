import cv2
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import time
import math
import random
import showGesture as extra
import numpy as np


triangle_points = {'name': 'triangle', 'points': [(137, 139), (135, 141), (133, 144), (132, 146), (130, 149), (128, 151), (126, 155), (123, 160), (120, 166), (116, 171), (112, 177), (107, 183), (102, 188), (100, 191), (95, 195), (90, 199), (86, 203), (82, 206), (80, 209), (75, 213), (73, 213), (70, 216), (67, 219), (64, 221), (61, 223), (60, 225), (62, 226), (65, 225), (67, 226), (74, 226), (77, 227), (85, 229), (91, 230), (99, 231), (108, 232), (116, 233), (125, 233), (134, 234), (145, 233), (153, 232), (160, 233), (170, 234), (177, 235), (179, 236), (186, 237), (193, 238), (198, 239), (200, 237), (202, 239), (204, 238), (206, 234), (205, 230), (202, 222), (197, 216), (192, 207), (186, 198), (179, 189), (174, 183), (170, 178), (164, 171), (161, 168), (154, 160), (148, 155), (143, 150), (138, 148), (136, 148)]}
x_points = {'name': 'x', 'points': [(87, 142), (89, 145), (91, 148), (93, 151), (96, 155), (98, 157), (100, 160), (102, 162), (106, 167), (108, 169), (110, 171), (115, 177), (119, 183), (123, 189), (127, 193), (129, 196), (133, 200), (137, 206), (140, 209), (143, 212), (146, 215), (151, 220), (153, 222), (155, 223), (157, 225), (158, 223), (157, 218), (155, 211), (154, 208), (152, 200), (150, 189), (148, 179), (147, 170), (147, 158), (147, 148), (147, 141), (147, 136), (144, 135), (142, 137), (140, 139), (135, 145), (131, 152), (124, 163), (116, 177), (108, 191), (100, 206), (94, 217), (91, 222), (89, 225), (87, 226), (87, 224)]}
rectangle_points = {'name': 'rectangle', 'points': [(78, 149), (78, 153), (78, 157), (78, 160), (79, 162), (79, 164), (79, 167), (79, 169), (79, 173), (79, 178), (79, 183), (80, 189), (80, 193), (80, 198), (80, 202), (81, 208), (81, 210), (81, 216), (82, 222), (82, 224), (82, 227), (83, 229), (83, 231), (85, 230), (88, 232), (90, 233), (92, 232), (94, 233), (99, 232), (102, 233), (106, 233), (109, 234), (117, 235), (123, 236), (126, 236), (135, 237), (142, 238), (145, 238), (152, 238), (154, 239), (165, 238), (174, 237), (179, 236), (186, 235), (191, 235), (195, 233), (197, 233), (200, 233), (201, 235), (201, 233), (199, 231), (198, 226), (198, 220), (196, 207), (195, 195), (195, 181), (195, 173), (195, 163), (194, 155), (192, 145), (192, 143), (192, 138), (191, 135), (191, 133), (191, 130), (190, 128), (188, 129), (186, 129), (181, 132), (173, 131), (162, 131), (151, 132), (149, 132), (138, 132), (136, 132), (122, 131), (120, 131), (109, 130), (107, 130), (90, 132), (81, 133), (76, 133)]}
circle_points = {'name': 'circle', 'points': [(127, 141), (124, 140), (120, 139), (118, 139), (116, 139), (111, 140), (109, 141), (104, 144), (100, 147), (96, 152), (93, 157), (90, 163), (87, 169), (85, 175), (83, 181), (82, 190), (82, 195), (83, 200), (84, 205), (88, 213), (91, 216), (96, 219), (103, 222), (108, 224), (111, 224), (120, 224), (133, 223), (142, 222), (152, 218), (160, 214), (167, 210), (173, 204), (178, 198), (179, 196), (182, 188), (182, 177), (178, 167), (170, 150), (163, 138), (152, 130), (143, 129), (140, 131), (129, 136), (126, 139)]}
check_points = {'name': 'check', 'points': [(91, 185), (93, 185), (95, 185), (97, 185), (100, 188), (102, 189), (104, 190), (106, 193), (108, 195), (110, 198), (112, 201), (114, 204), (115, 207), (117, 210), (118, 212), (120, 214), (121, 217), (122, 219), (123, 222), (124, 224), (126, 226), (127, 229), (129, 231), (130, 233), (129, 231), (129, 228), (129, 226), (129, 224), (129, 221), (129, 218), (129, 212), (129, 208), (130, 198), (132, 189), (134, 182), (137, 173), (143, 164), (147, 157), (151, 151), (155, 144), (161, 137), (165, 131), (171, 122), (174, 118), (176, 114), (177, 112), (177, 114), (175, 116), (173, 118)]}
caret_points = {'name': 'caret', 'points': [(79, 245), (79, 242), (79, 239), (80, 237), (80, 234), (81, 232), (82, 230), (84, 224), (86, 220), (86, 218), (87, 216), (88, 213), (90, 207), (91, 202), (92, 200), (93, 194), (94, 192), (96, 189), (97, 186), (100, 179), (102, 173), (105, 165), (107, 160), (109, 158), (112, 151), (115, 144), (117, 139), (119, 136), (119, 134), (120, 132), (121, 129), (122, 127), (124, 125), (126, 124), (129, 125), (131, 127), (132, 130), (136, 139), (141, 154), (145, 166), (151, 182), (156, 193), (157, 196), (161, 209), (162, 211), (167, 223), (169, 229), (170, 231), (173, 237), (176, 242), (177, 244), (179, 250), (181, 255), (182, 257)]}
arrow_points = {'name': 'arrow', 'points': [(68, 222), (70, 220), (73, 218), (75, 217), (77, 215), (80, 213), (82, 212), (84, 210), (87, 209), (89, 208), (92, 206), (95, 204), (101, 201), (106, 198), (112, 194), (118, 191), (124, 187), (127, 186), (132, 183), (138, 181), (141, 180), (146, 178), (154, 173), (159, 171), (161, 170), (166, 167), (168, 167), (171, 166), (174, 164), (177, 162), (180, 160), (182, 158), (183, 156), (181, 154), (178, 153), (171, 153), (164, 153), (160, 153), (150, 154), (147, 155), (141, 157), (137, 158), (135, 158), (137, 158), (140, 157), (143, 156), (151, 154), (160, 152), (170, 149), (179, 147), (185, 145), (192, 144), (196, 144), (198, 144), (200, 144), (201, 147), (199, 149), (194, 157), (191, 160), (186, 167), (180, 176), (177, 179), (171, 187), (169, 189), (165, 194), (164, 196)]}
left_square_bracket_points = {'name': 'left square bracket', 'points': [(140, 124), (138, 123), (135, 122), (133, 123), (130, 123), (128, 124), (125, 125), (122, 124), (120, 124), (118, 124), (116, 125), (113, 125), (111, 125), (108, 124), (106, 125), (104, 125), (102, 124), (100, 123), (98, 123), (95, 124), (93, 123), (90, 124), (88, 124), (85, 125), (83, 126), (81, 127), (81, 129), (82, 131), (82, 134), (83, 138), (84, 141), (84, 144), (85, 148), (85, 151), (86, 156), (86, 160), (86, 164), (86, 168), (87, 171), (87, 175), (87, 179), (87, 182), (87, 186), (88, 188), (88, 195), (88, 198), (88, 201), (88, 207), (89, 211), (89, 213), (89, 217), (89, 222), (88, 225), (88, 229), (88, 231), (88, 233), (88, 235), (89, 237), (89, 240), (89, 242), (91, 241), (94, 241), (96, 240), (98, 239), (105, 240), (109, 240), (113, 239), (116, 240), (121, 239), (130, 240), (136, 237), (139, 237), (144, 238), (151, 237), (157, 236), (159, 237)]}
right_square_bracket_points = {'name': 'right square bracket', 'points': [(112, 138), (112, 136), (115, 136), (118, 137), (120, 136), (123, 136), (125, 136), (128, 136), (131, 136), (134, 135), (137, 135), (140, 134), (143, 133), (145, 132), (147, 132), (149, 132), (152, 132), (153, 134), (154, 137), (155, 141), (156, 144), (157, 152), (158, 161), (160, 170), (162, 182), (164, 192), (166, 200), (167, 209), (168, 214), (168, 216), (169, 221), (169, 223), (169, 228), (169, 231), (166, 233), (164, 234), (161, 235), (155, 236), (147, 235), (140, 233), (131, 233), (124, 233), (117, 235), (114, 238), (112, 238)]}
v_points = {'name': 'v', 'points': [(89, 164), (90, 162), (92, 162), (94, 164), (95, 166), (96, 169), (97, 171), (99, 175), (101, 178), (103, 182), (106, 189), (108, 194), (111, 199), (114, 204), (117, 209), (119, 214), (122, 218), (124, 222), (126, 225), (128, 228), (130, 229), (133, 233), (134, 236), (136, 239), (138, 240), (139, 242), (140, 244), (142, 242), (142, 240), (142, 237), (143, 235), (143, 233), (145, 229), (146, 226), (148, 217), (149, 208), (149, 205), (151, 196), (151, 193), (153, 182), (155, 172), (157, 165), (159, 160), (162, 155), (164, 150), (165, 148), (166, 146)]}
delete_points = {'name': 'delete', 'points': [(123, 129), (123, 131), (124, 133), (125, 136), (127, 140), (129, 142), (133, 148), (137, 154), (143, 158), (145, 161), (148, 164), (153, 170), (158, 176), (160, 178), (164, 183), (168, 188), (171, 191), (175, 196), (178, 200), (180, 202), (181, 205), (184, 208), (186, 210), (187, 213), (188, 215), (186, 212), (183, 211), (177, 208), (169, 206), (162, 205), (154, 207), (145, 209), (137, 210), (129, 214), (122, 217), (118, 218), (111, 221), (109, 222), (110, 219), (112, 217), (118, 209), (120, 207), (128, 196), (135, 187), (138, 183), (148, 167), (157, 153), (163, 145), (165, 142), (172, 133), (177, 127), (179, 127), (180, 125)]}
left_curly_brace_points = {'name': 'left curly brace', 'points': [(150, 116), (147, 117), (145, 116), (142, 116), (139, 117), (136, 117), (133, 118), (129, 121), (126, 122), (123, 123), (120, 125), (118, 127), (115, 128), (113, 129), (112, 131), (113, 134), (115, 134), (117, 135), (120, 135), (123, 137), (126, 138), (129, 140), (135, 143), (137, 144), (139, 147), (141, 149), (140, 152), (139, 155), (134, 159), (131, 161), (124, 166), (121, 166), (117, 166), (114, 167), (112, 166), (114, 164), (116, 163), (118, 163), (120, 162), (122, 163), (125, 164), (127, 165), (129, 166), (130, 168), (129, 171), (127, 175), (125, 179), (123, 184), (121, 190), (120, 194), (119, 199), (120, 202), (123, 207), (127, 211), (133, 215), (142, 219), (148, 220), (151, 221)]}
right_curly_brace_points = {'name': 'right curly brace', 'points': [(117, 132), (115, 132), (115, 129), (117, 129), (119, 128), (122, 127), (125, 127), (127, 127), (130, 127), (133, 129), (136, 129), (138, 130), (140, 131), (143, 134), (144, 136), (145, 139), (145, 142), (145, 145), (145, 147), (145, 149), (144, 152), (142, 157), (141, 160), (139, 163), (137, 166), (135, 167), (133, 169), (131, 172), (128, 173), (126, 176), (125, 178), (125, 180), (125, 182), (126, 184), (128, 187), (130, 187), (132, 188), (135, 189), (140, 189), (145, 189), (150, 187), (155, 186), (157, 185), (159, 184), (156, 185), (154, 185), (149, 185), (145, 187), (141, 188), (136, 191), (134, 191), (131, 192), (129, 193), (129, 195), (129, 197), (131, 200), (133, 202), (136, 206), (139, 211), (142, 215), (145, 220), (147, 225), (148, 231), (147, 239), (144, 244), (139, 248), (134, 250), (126, 253), (119, 253), (115, 253)]}
star_points = {'name': 'star', 'points': [(75, 250), (75, 247), (77, 244), (78, 242), (79, 239), (80, 237), (82, 234), (82, 232), (84, 229), (85, 225), (87, 222), (88, 219), (89, 216), (91, 212), (92, 208), (94, 204), (95, 201), (96, 196), (97, 194), (98, 191), (100, 185), (102, 178), (104, 173), (104, 171), (105, 164), (106, 158), (107, 156), (107, 152), (108, 145), (109, 141), (110, 139), (112, 133), (113, 131), (116, 127), (117, 125), (119, 122), (121, 121), (123, 120), (125, 122), (125, 125), (127, 130), (128, 133), (131, 143), (136, 153), (140, 163), (144, 172), (145, 175), (151, 189), (156, 201), (161, 213), (166, 225), (169, 233), (171, 236), (174, 243), (177, 247), (178, 249), (179, 251), (180, 253), (180, 255), (179, 257), (177, 257), (174, 255), (169, 250), (164, 247), (160, 245), (149, 238), (138, 230), (127, 221), (124, 220), (112, 212), (110, 210), (96, 201), (84, 195), (74, 190), (64, 182), (55, 175), (51, 172), (49, 170), (51, 169), (56, 169), (66, 169), (78, 168), (92, 166), (107, 164), (123, 161), (140, 162), (156, 162), (171, 160), (173, 160), (186, 160), (195, 160), (198, 161), (203, 163), (208, 163), (206, 164), (200, 167), (187, 172), (174, 179), (172, 181), (153, 192), (137, 201), (123, 211), (112, 220), (99, 229), (90, 237), (80, 244), (73, 250), (69, 254), (69, 252)]}
pigtail_points = {'name': 'pigtail', 'points': [(81, 219), (84, 218), (86, 220), (88, 220), (90, 220), (92, 219), (95, 220), (97, 219), (99, 220), (102, 218), (105, 217), (107, 216), (110, 216), (113, 214), (116, 212), (118, 210), (121, 208), (124, 205), (126, 202), (129, 199), (132, 196), (136, 191), (139, 187), (142, 182), (144, 179), (146, 174), (148, 170), (149, 168), (151, 162), (152, 160), (152, 157), (152, 155), (152, 151), (152, 149), (152, 146), (149, 142), (148, 139), (145, 137), (141, 135), (139, 135), (134, 136), (130, 140), (128, 142), (126, 145), (122, 150), (119, 158), (117, 163), (115, 170), (114, 175), (117, 184), (120, 190), (125, 199), (129, 203), (133, 208), (138, 213), (145, 215), (155, 218), (164, 219), (166, 219), (177, 219), (182, 218), (192, 216), (196, 213), (199, 212), (201, 211)]}

images = ["arrow.jpg", "caret.jpg", "check.jpg", "circle.jpg", "delete.jpg", "left curly brace.jpg", "left square bracket.jpg", "pigtail.jpg", "rectangle.jpg", "right curly brace.jpg", "star.jpg", "triangle.jpg", "v.jpg", "x.jpg", "zig-zag.jpg"]
sample_images = ["rectangle.jpg", "left_square_bracket.jpg", "delete.jpg"]

original_gestures = [star_points, triangle_points, x_points, rectangle_points, circle_points, caret_points, arrow_points, left_square_bracket_points, right_square_bracket_points, v_points, delete_points, left_curly_brace_points, right_curly_brace_points, pigtail_points]


start_circle_x = 300
start_circle_y = 100
start_circle_r = 30

end_circle_x = 100
end_circle_y = 100
end_circle_r = 30

#Creates and returns a box that matches the leftmost, rightmost, topmost and bottommost points from the list points
def createBoundingBox(points):

    left = 1000
    right = -1000
    top = 1000
    bottom = -1000


    for p in points:

        if p[0] < left:
            left = p[0]

        if p[0] > right:
            right = p[0]

        if p[1] < top:
            top = p[1]

        if p[1] > bottom:
            bottom = p[1]

    return (left, top, right, bottom)


#Normalization and distance calculation
#Uses the distance function to calculate the distance between two points
def calc_distance(prev_point, curr_point):


    return math.sqrt(pow(abs(curr_point[0] - prev_point[0]),2) + pow(abs(curr_point[1] - prev_point[1]),2))


#Sums up the overall distance between each point in the list of points
def calculateOverallLength(points):

    length = 0.0

    for i in range(1, len(points)):
        length += calc_distance(points[i - 1], points[i])


    return length

#Resamples points
def resample(points):

    total_length = calculateOverallLength(points)
    threshold = total_length / 149

    resampled_points = [points[0]]
    current_path_length = 0.0

    for i in range(1, len(points)):
        prevPoint = points[i - 1]
        currPoint = points[i]
        distance = calc_distance(prevPoint, currPoint)

        while (current_path_length + distance) >= threshold:

            ratio = (threshold - current_path_length) / distance
            newX = prevPoint[0] + ratio * (currPoint[0] - prevPoint[0])
            newY = prevPoint[1] + ratio * (currPoint[1] - prevPoint[1])

            resampled_points.append((newX, newY))

            prevPoint = (newX, newY)
            distance -= (threshold - current_path_length)
            current_path_length = 0.0

        current_path_length += distance

    if len(resampled_points) < 150:
        resampled_points.append(points[-1])


    return resampled_points


#Moves points to the leftmost/topmost positions possible
def translate(points):

    left = 1000
    top = 1000
    translated_points = []

    for p in points:

        if (p[0] < left):
            left = p[0]
            leftMost = p

        if (p[1] < top):
            top = p[1]
            topMost = p


    for p in points:

        newPoint = (p[0] - leftMost[0], p[1] - topMost[1])
        translated_points.append(newPoint)


    return translated_points

#Uses the bounding box of the list points to scale the gesture downwards
def scale(points):

    left = createBoundingBox(points)[0]
    top = createBoundingBox(points)[1]
    right = createBoundingBox(points)[2]
    bottom = createBoundingBox(points)[3]

    scaledPoints = []

    if (right - left == 0):
        print(right)
        print(left)

    scaleX = 200 / (right - left)
    scaleY = 200 / (bottom - top)

    for p in points:
        scaledX = int(p[0] * scaleX)
        scaledY = int(p[1] * scaleY)
        newPoint = (scaledX, scaledY)

        scaledPoints.append(newPoint)

    return scaledPoints

#Checks if index finger is up
def index_finger_up(landmarks):

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    return index_tip.y < index_mcp.y

#Checks if pinky is up
def pinky_up(landmarks):

    index_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    index_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    index_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    return index_tip.y < index_mcp.y


#Checks if a fist is detected
def is_fist(landmarks):

    for finger_tip, finger_dip, finger_pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
    ]:
        if landmarks[finger_tip].y < landmarks[finger_pip].y:
            return False

    return True


def distance_between_thumb_and_index(landmarks, frame):

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    index_x, index_y = index_tip.x * frame.shape[1], index_tip.y * frame.shape[0]
    thumb_x, thumb_y = thumb_tip.x * frame.shape[1], thumb_tip.y * frame.shape[0]

    dist = calc_distance((index_x, index_y), (thumb_x, thumb_y))

    return dist




#Normalizes the given gestures using the methods resampled, translated and scaled
def normalize(gesture):

    resampled = resample(gesture)
    translated = translate(resampled)
    scaled = scale(translated)

    return scaled

#Converts the original template gesture points to normalized gestures
def store_normalized_gestures(original_gestures):
    normalized_gestures = []

    for g in original_gestures:
        normalized = normalize(g['points'])

        gesture_dict = {'name': g['name'], 'points': normalized}

        normalized_gestures.append(gesture_dict)


    return normalized_gestures

#Takes in a template gesture and a user entered gestures and finds the error distance between them
def compare(template_g, normalized):

    sum = 0
    min = 1000

    if (len(normalized) < min):
        min = len(normalized)

    if (len(template_g) < min):
        min = len(template_g)

    for i in range(min-1):

        sum += calc_distance(normalized[i], template_g[i])

    return sum



#Loops through all normalized template gestures and finds the template gesture with the smallest error distance with the user entered gesture
def findBestMatch(g, normalized_gestures):

    min = 1000000000000000
    normalized = normalize(g)
    # print("Best match called")
    # extra.calc_confidence(g)

    for template_g in normalized_gestures:

        num = len(template_g['points'])

        template_points = template_g['points']
        sum = compare(template_points, normalized)
        avg = sum / num
        #print('error threshold for ' + template_g['name'] + " - " + str(sum))
        if sum < min:
            min = sum
            best_match = template_g

    #print("\n\n\n")
    #print(best_match['name'] + " with an error threshold of " + str(min))
    return best_match

#Checks if (x,y) are inside (start_circle_x, start_circle_y) for a circle with radius start_circle_radius
def inside(x,y, start_circle_x, start_circle_y, start_circle_radius):

    if (x >= (start_circle_x - start_circle_r) and (x <= start_circle_x + start_circle_r)) and (y >= start_circle_y - start_circle_r) and (y <= start_circle_y + start_circle_r):
        return True
    else:
        return False


normalized_gestures = store_normalized_gestures(original_gestures)
image_display_counter = {"rectangle": 0, "left_square_bracket": 0, "delete": 0}

#Practice attempts
def practice():

    for i in range(3):

        image_path = random.choice(sample_images)

        hands = mp_hands.Hands(

            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        gesture_path = []

        cam = cv.VideoCapture(0)

        drawing_gesture = False
        end_gesture = False

        while cam.isOpened():

            success, frame = cam.read()

            if not success:
                print("Camera not available")
                continue

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            #image = cv.imread(image_path, cv.IMREAD_COLOR)
            cv.imshow("Show Gesture", frame)

            hands_detected = hands.process(frame)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            if hands_detected.multi_hand_landmarks:

                for hand_landmarks in hands_detected.multi_hand_landmarks:

                    landmarks = [lm for lm in hand_landmarks.landmark]

                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                    index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])


                    cv.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)


                    dist = distance_between_thumb_and_index(landmarks, frame)

                    if (dist <= 20):
                        gesture_path.clear()
                        drawing_gesture = True

                    if (dist >= 120):
                        end_gesture = True
                        drawing_gesture = False

                    if drawing_gesture:
                        if index_finger_up(landmarks):
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                            gesture_path.append((x, y))

                            for g in gesture_path:
                                cv.circle(frame, g, 3, (0, 0, 255), -1)

                if end_gesture:
                    break

            if cv.waitKey(20) & 0xff == ord('q'):
                break

##WORKS
# def showDrawnGesture(gesture_path, normalized_gestures):
#
#     if (len(gesture_path) > 0):
#         cam = cv.VideoCapture(0)
#
#         while (cam.isOpened()):
#             success, frame = cam.read()
#
#             if not success:
#                 print("Camera not available")
#                 continue
#
#             frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
#
#             normalized = normalize(gesture_path)
#
#             for p in normalized:
#                 cv.circle(frame, p, 3, (255, 255, 255, -1))
#
#             o_normalized = findBestMatch(gesture_path, normalized_gestures)['points']
#
#             for o1 in o_normalized:
#                 cv.circle(frame, o1, 3, (255, 0, 0, -1))
#
#             cv.imshow("Show Video", frame)
#
#             if cv.waitKey(20) & 0xff == ord('q'):
#                 break
#
#         cam.release()

def showDrawnGesture(gesture_path, normalized_gestures):

    if (len(gesture_path) > 0):
        cam = cv.VideoCapture(0)

        while (cam.isOpened()):
            success, frame = cam.read()

            if not success:
                print("Camera not available")
                continue

            frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)

            normalized = normalize(gesture_path)

            # for p in normalized:
            #     cv.circle(frame, p, 3, (255, 255, 255, -1))

            o_normalized_name = findBestMatch(gesture_path, normalized_gestures)['name']

            for img in sample_images:
                if img.removesuffix(".jpg") == o_normalized_name:
                    to_show = img

            # for o1 in o_normalized:
            #     cv.circle(frame, o1, 3, (255, 0, 0, -1))

            #cv.imshow("Show Video", o_normalized_name)

            # image = cv.imread(to_show, cv.IMREAD_COLOR)
            # cv.imshow("Show Gesture", image)
            image = cv.imread(to_show, cv.IMREAD_COLOR)

            # # Add "Recognized:" label
            # label_bg_color = (255, 255, 255)  # White background for the label
            # text_color = (0, 0, 0)  # Black text
            # font_scale = 0.3
            # thickness = 1
            #
            # # Get text size
            # (text_width, text_height), _ = cv.getTextSize("Recognized:", cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            #
            # # Define rectangle coordinates
            # rect_x1, rect_y1 = 25, 0  # Top-left corner of the label
            # rect_x2, rect_y2 = rect_x1 + text_width + 5, rect_y1 + text_height + 3  # Bottom-right corner
            #
            # # Draw label background (rectangle)
            # cv.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), label_bg_color, -1)
            #
            # # Add text inside the rectangle
            # cv.putText(image, "Recognized:", (rect_x1 + 5, rect_y2 - 5), cv.FONT_HERSHEY_SIMPLEX, font_scale,
            #            text_color, thickness, cv.LINE_AA)
            #
            # # Create a full-screen window
            # cv.namedWindow("Show Gesture", cv.WINDOW_NORMAL)
            #
            # #cv2.putText(frame, "Elapsed Time - " + str(round(elapsed_time, 2)), (50, 50), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            # #cv.putText(image, "Recognized:", (5, 10), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 2)
            #
            # cv.resizeWindow("Show Gesture", 600, 600)
            #
            # cv.imshow("Show Gesture", image)

            image = cv.resize(image, (800, 600))  # Resize gesture image for clarity

            # Create a label/header (white background)
            label_height = 60
            label_width = image.shape[1]  # Match image width
            label_frame = np.ones((label_height, label_width, 3), dtype=np.uint8) * 255  # White background

            # Add text to the label frame
            text = "Recognized Gesture:"
            font_scale = 1
            thickness = 2
            text_color = (0, 0, 0)  # Black text
            (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Position text at the center of the label
            text_x = (label_width - text_width) // 2
            text_y = (label_height + text_height) // 2
            cv.putText(label_frame, text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness,
                       cv.LINE_AA)

            # Stack the label frame on top of the gesture image
            combined_image = np.vstack((label_frame, image))

            # Create a resizable window
            cv.namedWindow("Gesture Recognition", cv.WINDOW_NORMAL)
            cv.resizeWindow("Gesture Recognition", label_width, label_height + image.shape[0])

            cv.imshow("Gesture Recognition", combined_image)



            if cv.waitKey(20) & 0xff == ord('q'):
                break

        cam.release()


def gesture_test(enter_counter):
    drawing_prompt = False
    end_prompt = False

    image_path = random.choice(sample_images)
    given_gesture = image_path.removesuffix(".jpg")

    while (image_display_counter[given_gesture] == 10):
        image_path = random.choice(sample_images)
        given_gesture = image_path.removesuffix(".jpg")

    image_display_counter[given_gesture] += 1

    hands = mp_hands.Hands(

        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    gesture_path = []

    cam = cv.VideoCapture(0)

    drawing_gesture = False
    end_gesture = False

    while cam.isOpened():

        success, frame = cam.read()

        if not success:
            print("Camera not available")
            continue

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        cv.imshow("Show Gesture", image)

        hands_detected = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if hands_detected.multi_hand_landmarks:

            for hand_landmarks in hands_detected.multi_hand_landmarks:

                landmarks = [lm for lm in hand_landmarks.landmark]

                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                cv.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)

                dist = distance_between_thumb_and_index(landmarks, frame)

                if dist <= 20 and not drawing_gesture:
                    print("Pinch recognized")
                    gesture_path.clear()
                    drawing_gesture = True

                if dist >= 120 and drawing_gesture:
                    print("Gesture recording stopped")
                    drawing_gesture = False
                    enter_counter = enter_counter + 1
                    print(enter_counter)

                    if (len(gesture_path) == 0):
                        print("No gesture detected")
                        f = open("userresults.txt", "a")
                        f.write(str(enter_counter))
                        f.write("\n")
                        f.write("No gesture detected")
                        f.write("\n")
                        f.write("\n")
                        f.close()
                    else:
                        best_match = findBestMatch(gesture_path, normalized_gestures)
                        conf = extra.calc_confidence(gesture_path, best_match['name'])
                        print("Intended Gesture - " + given_gesture)
                        print("Recognized Gesture - " + best_match['name'] + " (" + str(round(conf, 1)) + ")")
                        print(gesture_path)

                        f = open("userresults.txt", "a")
                        f.write(str(enter_counter))
                        f.write("\n")
                        f.write("Intended Gesture - " + given_gesture)
                        f.write("\n")
                        f.write("Recognized Gesture - " + best_match['name'] + " (" + str(round(conf,1)) + ")")
                        f.write("\n")
                        f.write(str(gesture_path))
                        f.write("\n")
                        f.write("\n")
                        f.close()

                    if enter_counter != 31:
                        showDrawnGesture(gesture_path, normalized_gestures)
                        gesture_path.clear()
                        gesture_test(enter_counter)
                    else:
                        cam.release()

                if drawing_gesture:
                    if index_finger_up(landmarks):
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                        gesture_path.append((x, y))


        if cv.waitKey(20) & 0xff == ord('q'):
            break


        if (cv.waitKey(100) & 0xff == 13):

            print("Pressed Enter")
            # enter_counter = enter_counter + 1
            # print(enter_counter)
            #
            # if (len(gesture_path) == 0):
            #     print("No gesture detected")
            # else:
            #     best_match = findBestMatch(gesture_path, normalized_gestures)
            #     print("Intended Gesture - " + given_gesture)
            #     print("Drawn Gesture - " + best_match['name'])
            #     print(gesture_path)
            #
            # if (enter_counter != 30):
            #     showDrawnGesture(gesture_path, normalized_gestures)
            #     gesture_test(enter_counter)
            # else:
            #     cam.release()


def showTemplate(image_path):

    image = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow("Show Gesture", image)
    return

def trial(given_gesture):

    gesture_path = []
    cam = cv.VideoCapture(0)

    hands = mp_hands.Hands(

        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    drawingGesture = False
    end = False

    while cam.isOpened():

        success, frame = cam.read()

        if not success:
            print("Camera not available")
            continue

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        hands_detected = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if hands_detected.multi_hand_landmarks:

            for hand_landmarks in hands_detected.multi_hand_landmarks:

                landmarks = [lm for lm in hand_landmarks.landmark]

                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                dist = distance_between_thumb_and_index(landmarks, frame)

                if dist <= 15 and (not drawingGesture):
                    print("Pinch recognized")
                    drawingGesture = True
                elif dist >= 120 and drawingGesture:
                    print("Gesture ending recognizer")

                    if (len(gesture_path) == 0):
                        print("No gesture detected")
                    else:
                        best_match = findBestMatch(gesture_path, normalized_gestures)
                        conf = extra.calc_confidence(gesture_path, best_match['name'])
                        print("Intended Gesture - " + given_gesture)
                        print("Recognized Gesture - " + best_match['name'] + " (" + str(round(conf, 4)) + ")")
                        print(gesture_path)
                        showDrawnGesture(gesture_path, normalized_gestures)
                        end = True
                        break

                if drawingGesture:
                    if index_finger_up(landmarks):
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                        gesture_path.append((x, y))

                        # for point in gesture_path:
                        #     cv.circle(frame, point, 3, (0, 255, 0, -1))


        cv.imshow("Show Video", frame)
        if (end):
            return

        if cv.waitKey(20) & 0xff == ord('q'):
            break

        if (cv.waitKey(20) & 0xff == ord('c')):
            gesture_path.clear()

    cam.release()


def showPath(gesture_path):
    enter_pressed = False

    if (len(gesture_path) > 0):

        image = cv.imread("black.jpg", cv.IMREAD_COLOR)

        for p in gesture_path:

            cv.circle(image, p, 10, (255, 255, 255, 255), -1)

            if cv.waitKey(20) & 0xff == ord('q'):
                break

        while (not enter_pressed):
            cv.imshow("Show Gesture", image)

            key = cv.waitKey(20) & 0xFF
            if key == 13:
                enter_pressed = True



def gesture(given_gesture):

    drawingGesture = False

    hands = mp_hands.Hands(

        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    gesture_path = []

    cam = cv.VideoCapture(0)

    while cam.isOpened():

        success, frame = cam.read()
        cv.imshow("Show Video", frame)

        if not success:
            print("Camera not available")
            continue

        hands_detected = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if hands_detected.multi_hand_landmarks:

            for hand_landmarks in hands_detected.multi_hand_landmarks:

                landmarks = [lm for lm in hand_landmarks.landmark]

                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style(),
                )

                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                dist = distance_between_thumb_and_index(landmarks, frame)

                if dist <= 15 and (not drawingGesture):
                    print("Pinch recognized")
                    drawingGesture = True
                elif dist >= 120 and drawingGesture:
                    print("Gesture ending recognizer")

                    if (len(gesture_path) == 0):
                        print("No gesture detected")
                    else:
                        best_match = findBestMatch(gesture_path, normalized_gestures)
                        print("Intended Gesture - " + given_gesture)
                        print("Drawn Gesture - " + best_match['name'])
                        print(gesture_path)
                        showPath(gesture_path)
                        cam.release()
                        return 1


                if drawingGesture:
                    if index_finger_up(landmarks):
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                        gesture_path.append((x, y))

                        for point in gesture_path:
                            cv.circle(frame, point, 3, (0, 255, 0, -1))

# for i in range(31):
#
#     print(i+1)
#     image_path = random.choice(sample_images)
#     given_gesture = image_path.removesuffix(".jpg")
#     while (image_display_counter[given_gesture] == 10):
#         image_path = random.choice(sample_images)
#         given_gesture = image_path.removesuffix(".jpg")
#
#
#     image = cv.imread(image_path, cv.IMREAD_COLOR)
#     cv.imshow("Show Template", image)
#
#     image_display_counter[given_gesture] += 1
#
#     #trial(given_gesture)
#     response = gesture(given_gesture)
#     # print("Out of there")


enter_counter = 0
#gesture_test(enter_counter)
trial("delete")














# best_match = findBestMatch(gesture_path, normalized_gestures)
# print("*********************************************************************************")
# print(gesture_path)


















