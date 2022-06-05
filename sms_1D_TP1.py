#This is a python application that performs the operation feature extraction on a given SMS message using One-Dimensional Ternary Patterns Transformation

#Define a function to generate the UTF-8 value of the given SMS
def genTextUTF8(text):
	txtUTF8 = []
	for ch in text:
		txtUTF8.append(ord(ch))
	return txtUTF8


#Define a function to perform this operation
def genTextPatterns(text, i, c):
	textList = []
	#Convert the text to a list
	textList += text
	#Remove the ith element/character of the list
	a = textList.pop(i)
	#Add the removed into the middle/center of the list
	textList.insert(c, a)
	return textList


#Creating a function to convert binary to decimal
def binaryToDecimal(binaryList):
	decimal, n = 0, len(binaryList)	
	for b in binaryList:
		decimal += b*2**(n-1)
		n -= 1
	return decimal

#Define a function to extract the unique Histogram Value
def extractUniqueHistogramValues(valueList, histLength):
	uniqueHistoValueList = []
	for i in range(0, 2**histLength):
		if(i in valueList):
			uniqueHistoValueList.append(i)
	return uniqueHistoValueList

#Define a function to extract the frequency of occurrence of each unique Histogram Value in the value List
def extractHistogramFreq(valueList, uniqHistList):
	uniqueHistoFreqList = []
	for i in uniqHistList:
		uniqueHistoFreqList.append(valueList.count(i))
	return uniqueHistoFreqList		
		
	


import numpy as np
import matplotlib.pyplot as plt		
#Read the given SMS
#originalSMS = "A Novel Feature Extraction Approach in SMS Spam Filtering for Mobile Communication: One-Dimensional Ternary Patterns"
originalSMS = "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"
print("========================================== Text Result for Spam Message at P = 10 and B = 2  ======================================================\n")
print("Dataset: 'SMS Spam Corpus v.0.1.txt'\n")
print("The original sms (SPAM): '", originalSMS, "' with length: ", len(originalSMS))
#Remove all flux from the SMS
originalSMS = originalSMS.replace(" ","").replace("'","")
SMS_length = len(originalSMS)
#Set the initial value of P and the threshold value and perform the 1D-TP transformation
P, B = 10, 2 
#Get the number of possible pattern P that can be formed on the SMS
noOfPatterns = SMS_length//(P+1)
upFeaturesList, lowFeaturesList = [],[]
n, step = 1, P+1
for k in range(0, (noOfPatterns*step), step):
	text = originalSMS[k:n*step]
	print("Pattern ",n," Text is: ", text)
	for x in range(0, len(text)):
		#call the function to generate the pattern for the given text
		textPattern = genTextPatterns(text, x, len(text)//2)
		#Partition the text into 3 list(Pl, Pr, Pc)
		Lp, Cp, Rp = textPattern[:P//2], textPattern[P//2:P//2+1], textPattern[P//2+1:] 
		print("The left list is: ", Lp)
		print("The centre list is: ", Cp)	
		print("The right list is: ", Rp)
		#Get the UTF-8 of the characters in the given SMS text pattern	
		Pl, Pr, Pc = [],[], ord(Cp[0])
		for i in range(0,len(Lp)):
			Pl.insert(i, ord(Lp[i]))
		for i in range(0,len(Rp)):
			Pr.insert(i, ord(Rp[i]))
		print("The left list is: ", Pl)
		print("The centre list is: ", Pc)	
		print("The right list is: ", Pr)
		#Comparison of Pc with neighbors (Pi), 
		Tpl, Tpr = [],[]
		for i in range(0,len(Lp)):
			if(Pc > (Pl[i] + B)):		
				Tpl.insert(i, 1)
			elif(Pc <= (Pl[i] + B) and Pc >= (Pl[i] - B)):		
				Tpl.insert(i, 0)
			elif(Pc < (Pl[i] - B)):		
				Tpl.insert(i, -1)
		for i in range(0,len(Rp)):
			if(Pc > (Pr[i] + B)):		
				Tpr.insert(i, 1)
			elif(Pc <= (Pr[i] + B) and Pc >= (Pr[i] - B)):		
				Tpr.insert(i, 0)
			elif(Pc < (Pr[i] - B)):		
				Tpr.insert(i, -1)
		print("The left list is: ", Tpl)
		print("The right list is: ", Tpr)
		#Separation positive and negative values,	
		upF, lowF = [],[]
		for i in range(0,len(Lp)):
			if(Tpl[i] == -1):		
				upF.insert(i, 0)
				lowF.insert(i, 1)
			else:		
				upF.insert(i, Tpl[i])
				lowF.insert(i, 0)
		for i in range(0,len(Rp)):
			if(Tpr[i] == -1):		
				upF.insert(len(Lp)+i, 0)
				lowF.insert(len(Lp)+i, 1)
			else:		
				upF.insert(len(Lp)+i, Tpr[i])
				lowF.insert(len(Lp)+i, 0)
		print("The up list is: ", upF)
		print("The low list is: ", lowF)
		#Conversion of binary values to decimal	
		upFeatures = binaryToDecimal(upF)
		lowFeatures = binaryToDecimal(lowF)
		print("The upFeatures is: ", upFeatures)
		print("The lowFeatures is: ", lowFeatures)
		#Populate the both upFeatures List and lowFeaturesList
		upFeaturesList.append(upFeatures)
		lowFeaturesList.append(lowFeatures)
		#increase the value of n for the pattern to be selected for computation
	n += 1
#Define two separate lists for each of the up and low 1D-TP signal histogram values
upHistValues = extractUniqueHistogramValues(upFeaturesList, P)
lowHistValues = extractUniqueHistogramValues(lowFeaturesList, P)
upHistFreq = extractHistogramFreq(upFeaturesList, upHistValues)
lowHistFreq = extractHistogramFreq(lowFeaturesList, lowHistValues)
print("The preprocessed sms: '", originalSMS, "'\n")
print("The number of pattern is: ", noOfPatterns, " and value of P is: ", P, " with threshold value (B) is: ", B, " \n")
print("The UTF-8 values for the sms is: ", genTextUTF8(originalSMS), " \n")
print("The total number of Features for the sms is: ", 2**P, " \n")
print("The up Features 1D-TP signals for the sms is: ", upFeaturesList, " \n")
print("The total number of up Features for the sms is: ", len(upFeaturesList), " \n")
print("The low Features 1D-TP signals for the sms is: ", lowFeaturesList, " \n")
print("The total number of low Features for the sms is: ", len(lowFeaturesList), " \n")
print("The Up Features 1D-TP signals unique Histogram Values for the sms is: ", upHistValues, " \nwith frequency values: ", upHistFreq, " \n")
print("The Low Features 1D-TP signals unique Histogram Values for the sms is: ", lowHistValues, " \nwith frequency values: ", lowHistFreq)
#Plotting a graph to show the output of the UTF-8 of the SMS
fig1 = plt.figure(1)
x = np.arange(0, len(originalSMS))
y = genTextUTF8(originalSMS)
plt.plot(x, y)
#plt.title("Graph of Unicodes of a sample SMS Message at P = 8 and B = 3")
plt.title("Spam Message Unicodes Signal")
plt.ylabel("UTF-8 values of characters")
#Plotting a graph to show the output of the UPFeatures 1D-TP signals for the SMS
fig2 = plt.figure(2)
x = np.arange(0, len(upFeaturesList))
y = upFeaturesList
plt.plot(x, y)
plt.title("1D-TP Upper Features signal for SPAM Message at P = " + str(P) + " and B = " + str(B))
plt.ylabel("UP Features values")
#Plotting a graph to show the output of the lowFeatures 1D-TP signals for the SMS
fig2 = plt.figure(3)
x = np.arange(0, len(lowFeaturesList))
y = lowFeaturesList
plt.plot(x, y)
plt.title("1D-TP Lower Features signal for SPAM Message at P = " + str(P) + " and B = " + str(B))
plt.ylabel("Low Features values")
#Plotting a graph to show the output of the Histogram of the UPFeatures 1D-TP signals for the SMS
fig2 = plt.figure(4)
x = upHistValues
y = upHistFreq
plt.plot(x, y)
plt.title("1D-TP Histogram(Upper) for SPAM Message at P = " + str(P) + " and B = " + str(B))
plt.ylabel("Frequency")
plt.xlabel("Unique Value for Up Features")
#Plotting a graph to show the output of the Histogram of the LOWFeatures 1D-TP signals for the SMS
fig2 = plt.figure(5)
x = lowHistValues
y = lowHistFreq
plt.plot(x, y)
plt.title("1D-TP Histogram(Lower) for SPAM Message at P = " + str(P) + " and B = " + str(B))
plt.ylabel("Frequency")
plt.xlabel("Unique Value for Low Features")

#Show all the graphs
plt.show()
#Press any key to continue
input("Press any key to continue")





