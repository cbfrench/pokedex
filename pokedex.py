from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from PIL import Image
import pytesseract
from pytesseract import Output

class Card:
	def __init__(self, name):
		self.name = name
		self.movelist = []

class Move:
	def __init__(self, name, description):
		self.name = name
		self.damage = "0"
		self.description = description

def wait_input():
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#def clean_input(input):
#	ind = input.find(" ") + 1
#	return input[ind:]

def is_valid_character(input):
	a = ord(input) < 48
	b = ord(input) > 57 and ord(input) < 65
	c = ord(input) > 90 and ord(input) < 97
	d = ord(input) > 122
	return not (a or b or c or d)

def clean_input(input):
	result = ""
	inWord = False
	for s in input.split(" "):
		count = 0
		for c in s:
			if is_valid_character(c):
				count += 1
		if count * 2 > len(s):
			result += s + " "
	return result

def find_edges():
	#load image and get ratio of old height to new height, clone, then resize
	global image, orig, ratio
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	
	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	cv2.imshow("Image", image)
	cv2.imshow("Edged", edged)
	wait_input()
	return edged

def find_contours():
	global image, edged
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	
	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of card")
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	wait_input()
	return screenCnt

def warp():
	global screenCnt
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	#T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	#warped = (warped > T).astype("uint8") * 255
	return warped

def convert_to_grayscale(image):
	#convert move section to black and white
	im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = 120
	im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

	#cv2.imwrite("moves-bw.png", im_bw)
	#cv2.imshow("BW", im_bw)
	#wait_input()
	return im_bw

def find_moves(raw_move_text, text_boxes, energy_positions, move_names_text):
	print("STEP 7: Parse out the name of each move")
	adjusted_text_boxes = []
	adjusted_move_names = []
	move_text = raw_move_text.split('\n')
	move_names = move_names_text.split('\n')
	adjusted_move_text = []
	moves_index = []
	moves = []
	threshold = 20
	try:
		for a in text_boxes:
			if not any(t[1] == a[1] for t in adjusted_text_boxes):
				if a[1] == 0:
					continue
				adjusted_text_boxes.append(a)
		for i in range(0, len(adjusted_text_boxes)):
			for j in range(0, len(energy_positions)):
				a = adjusted_text_boxes[i][1] + threshold >= energy_positions[j][1]
				b = adjusted_text_boxes[i][1] - threshold <= energy_positions[j][1]
				if a and b:
					#print("Move found at line " + str(i))
					moves_index.append(i)
					break
		for i in range(0, len(move_text)):
			if move_text[i] != '':
				adjusted_move_text.append(clean_input(move_text[i]))
				adjusted_move_names.append(clean_input(move_names[i]))
		for s in moves_index:
			m = Move(adjusted_move_names[s], "")
			moves.append(m)
		count = 0
		description = ""
		for i in range(1, len(adjusted_move_text)):
			if i in moves_index:
				moves[count].description = description
				description = ""
				count += 1
				continue
			else:
				description += adjusted_move_text[i]
				#print(description)
		if description != "":
			moves[len(moves)-1].description = description
		return moves
	except:
		print("There was an error scanning your card, please try again.")
		return -1

def check_move_damage(moves):
	result = []
	for m in moves:
		s = m.name.split(" ")
		n = ""
		num = "0"
		for w in s:
			if any(char.isdigit() for char in w):
				num = w
			else:
				if n == "":
					n += w
				else:
					n += " " + w
		x = Move(n, m.description)
		x.damage = num
		result.append(x)
	return result

def get_energy_positions(img_rgb):
	#use black and white move section to find colorless energy requirements
	print("STEP 5: Find y values for all energies in move costs")
	template = cv2.imread('colorless.png')
	w, h = template.shape[:-1]

	res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
	threshold = .8
	loc = np.where(res >= threshold)
	count = 0
	p_thresh = 10
	result = []
	for pt in zip(*loc[::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
		found = False
		for p in result:
			horiz = pt[0] - p_thresh <= p[0] <= pt[0] + p_thresh
			vert = pt[1] - p_thresh <= p[1] <= pt[1] + p_thresh
			if horiz and vert:
				found = True
		if(not found):
			count = count + 1
			result.append(pt)
	return result

def get_text_boxes():
	print("STEP 6: Find y values for all text boxes")
	img = cv2.imread('moves.png')
	text_boxes = []
	d = pytesseract.image_to_data(img, output_type=Output.DICT)
	n_boxes = len(d['level'])
	for i in range(n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		if w * h > 80000:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			text_boxes.append([x, y, w, h])
			#print(str(x) + " " + str(y))

	#cv2.imshow('img', img)
	#cv2.imwrite('img-dev.png', img)
	#cv2.waitKey(0)
	return text_boxes 

def display_card(card):
	print("\n\n******POKEMON******")
	print(card.name + "\n")
	for m in card.movelist:
		print("MOVE")
		print("Name: " + m.name)
		print("Damage: " + m.damage)
		print("Description: " + m.description + '\n\n')
	final_card = cv2.imread("card.png")
	final_card = cv2.resize(final_card, (600, 800))
	cv2.imshow(card.name, final_card)
	wait_input()

def build_card():
	global warped
	# show the original and scanned images
	print("STEP 3: Apply perspective transform")
	#cv2.imshow("Original", imutils.resize(orig, height = 650))
	#cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	warped = cv2.resize(warped, (2100, 2600))
	gray_warped = convert_to_grayscale(warped)

	print("STEP 4: Get all relevant sections of the card")
	name_image = warped[int(warped.shape[0]/16 * 0.5):int(warped.shape[0]/16 * 1.35), int(warped.shape[1]/16 * 2.6):int(warped.shape[1]/16 * 9)]
	ability_image = warped[int(warped.shape[0]/16 * 8.7):int(warped.shape[0]/16 * 9.1), int(warped.shape[1]/16 * 1.3):int(warped.shape[1]/16 * 4.2)]
	move_names_image = warped[int(warped.shape[0]/16 * 8.25):int(warped.shape[0]/16 * 13.75), int(warped.shape[1]/16 * 5):int(warped.shape[1]/16 * 15.6)]
	moves_image = warped[int(warped.shape[0]/16 * 8.25):int(warped.shape[0]/16 * 13.75), int(warped.shape[1]/16 * 0.4):int(warped.shape[1]/16 * 15.6)]
	weakness_image = warped[int(warped.shape[0]/16 * 13.9):int(warped.shape[0]/16 * 14.4), int(warped.shape[1]/16 * 0.5):int(warped.shape[1]/16 * 4.5)]
	info_image = warped[int(warped.shape[0]/16 * 14.5):int(warped.shape[0]/16 * 15.75), int(warped.shape[1]/16 * 0.5):int(warped.shape[1]/16 * 4.5)]

	gray_name_image = gray_warped[int(warped.shape[0]/16 * 0.5):int(warped.shape[0]/16 * 1.35), int(warped.shape[1]/16 * 2.6):int(warped.shape[1]/16 * 9)]
	gray_ability_image = gray_warped[int(warped.shape[0]/16 * 8.7):int(warped.shape[0]/16 * 9.1), int(warped.shape[1]/16 * 1.3):int(warped.shape[1]/16 * 4.2)]
	gray_move_names_image = gray_warped[int(warped.shape[0]/16 * 8.25):int(warped.shape[0]/16 * 13.75), int(warped.shape[1]/16 * 5):int(warped.shape[1]/16 * 15.6)]
	gray_moves_image = gray_warped[int(warped.shape[0]/16 * 8.25):int(warped.shape[0]/16 * 13.75), int(warped.shape[1]/16 * 0.4):int(warped.shape[1]/16 * 15.6)]
	gray_weakness_image = gray_warped[int(warped.shape[0]/16 * 13.9):int(warped.shape[0]/16 * 14.4), int(warped.shape[1]/16 * 0.5):int(warped.shape[1]/16 * 4.5)]
	gray_info_image = gray_warped[int(warped.shape[0]/16 * 14.5):int(warped.shape[0]/16 * 15.75), int(warped.shape[1]/16 * 0.5):int(warped.shape[1]/16 * 4.5)]

	#col = warped[1755:1860, 105:220]
	#cv2.imshow('Col', col)
	#g_col = convert_to_grayscale(col)
	#cv2.imwrite('colorless.png', g_col)

	#cv2.imshow("Name", name_image)
	#cv2.imshow("Moves", moves_image)
	#cv2.imshow("Weakness", weakness_image)
	#cv2.imshow("Other Information", info_image)
	#wait_input()

	#imS = cv2.resize(warped, (1350, 1350))
	cv2.imwrite('card.png', warped)
	cv2.imwrite('name.png', name_image)
	cv2.imwrite('ability.png', ability_image)
	cv2.imwrite('move_names.png', move_names_image)
	cv2.imwrite('moves.png', moves_image)
	cv2.imwrite('weakness.png', weakness_image)
	cv2.imwrite('other-info.png', info_image)
	
	cv2.imwrite('gray_card.png', gray_warped)
	cv2.imwrite('gray_move_names.png', gray_move_names_image)
	cv2.imwrite('gray_ability.png', gray_ability_image)
	cv2.imwrite('gray_name.png', gray_name_image)
	cv2.imwrite('gray_moves.png', gray_moves_image)
	cv2.imwrite('gray_weakness.png', gray_weakness_image)
	cv2.imwrite('gray_other-info.png', gray_info_image)

	pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
	name_text = clean_input(pytesseract.image_to_string(name_image, lang="eng"))
	ability_text = clean_input(pytesseract.image_to_string(ability_image, lang="eng"))
	move_names_text = clean_input(pytesseract.image_to_string(move_names_image, lang="eng"))
	moves_text = clean_input(pytesseract.image_to_string(moves_image, lang="eng"))
	weakness_text = clean_input(pytesseract.image_to_string(weakness_image, lang="eng"))
	info_text = clean_input(pytesseract.image_to_string(info_image, lang="eng"))

	card = Card(name_text)


	#print("Name: " + name_text)
	#print("Ability: " + ability_text)
	#print("Move Names: " + move_names_text)
	#print("Moves: " + moves_text)
	#print("Weakness: " + weakness_text)
	#print("Additional Info: " + info_text)

	img_rgb = cv2.imread('gray_moves.png')
	energy_positions = get_energy_positions(img_rgb)

	#cv2.imwrite('result.png', img_rgb)
	#cv2.imshow("MATCHES", img_rgb)
	#wait_input()
	final_result = Card(name_text)
	final_result.movelist = []

	text_boxes = get_text_boxes()

	final_result.movelist = find_moves(moves_text, text_boxes, energy_positions, move_names_text)
	if(final_result.movelist != -1):
		final_result.movelist = check_move_damage(final_result.movelist)
		display_card(final_result)

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "Path to the image to be scanned")
	args = vars(ap.parse_args())

	orig = None
	ratio = None
	image = cv2.imread(args["image"])
	edged = find_edges()
	screenCnt = find_contours()
	warped = warp()
	pokemon = build_card()
	#print("\n")
	'''
	print("Pokemon: " + pokemon.name)
	count = 1
	for x in pokemon.movelist:
		print("Move " + str(count) + ": " + x.name)
		print("\t" + x.description)
		count += 1
	'''