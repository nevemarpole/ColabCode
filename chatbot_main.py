# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:03:25 2020

@author: nevem
"""

import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import random
from math import log10
import pickle

#Tokenizes, lowers the case, removes stop words, stems the text passed to it
def rework_text(text, check_value):
    
    #Tokenize the text to make it a list of lists
    #Allowing words to be accessed indivdually
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    changed_text = []
    for string in text:
        changed_text.append(tokenizer.tokenize(string))


    if check_value == True:
        #Remove stop words and make all charcters lower case
        documents = []
        english_stopwords = stopwords.words('english')
        for tok_document in changed_text:
            documents.append([word.lower() for word in tok_document if word not in english_stopwords])
        
        #Stem all words left
        sb_stemmer = SnowballStemmer('english')
        stemmed_documents = []
        for part in documents:
            stemmed_documents.append([sb_stemmer.stem(word) for word in part]) 
        changed_text = stemmed_documents
    
    return(changed_text)
     





#determin how similar a user question is to a standard file question
def get_similarity(user_text, a_question):

    #Store the user's question and the file question in a list
    #Accessed one after another by the following loops
    strings = [user_text, a_question] 
    
    #Create vocabulary for the user's question and the question from file
    vocabulary = []
    for string in strings:
        for item in string:
            #index = vocabulary.index(item)
            #vector[index] +=1
            if item not in vocabulary:
                vocabulary.append(item)
    
    #Create bag of words for the user's question and the question from file
    bow = []
    for string in strings:
        vector = np.zeros(len(vocabulary))
        for item in string:
            index = vocabulary.index(item)
            vector[index] += 1
        bow.append(vector)
        
    #Get TF-IDF, the multiple return values stored as a list
    result = tfidf_weight(bow[0], bow[1])

    #Get manhattan distance and use this to work out final similarity value
    distance = manhattan_distance(result[0], result[1])
    similarity = 1 / (1+distance)
    
    return similarity






#Works out the TF-IDF for both vectors passed to it
def tfidf_weight(vector_1, vector_2):
    
    N = 2
    tfidf_vector_1 = np.zeros(len(vector_1))
    tfidf_vector_2 = np.zeros(len(vector_2))
    
    for i in range(len(vector_1)):
        
        term_booleans = [vector_1[i]!=0, vector_2[i]!=0]
        n = sum(term_booleans)
        
        frequency_1 = vector_1[i]
        tfidf_1 = log10(1+frequency_1) * log10(N/n)
        tfidf_vector_1[i] = tfidf_1
        
        frequency_2 = vector_2[i]
        tfidf_2 = log10(1+frequency_2) * log10(N/n)
        tfidf_vector_2[i] = tfidf_2
        
    return tfidf_vector_1, tfidf_vector_2






#Works out the manhattan distance between 2 vectors
def manhattan_distance(vector_1, vector_2):
    
    distance = abs(vector_1 - vector_2)
    
    return distance.sum()













#Load the multinoial naive bayes model
nb_model = pickle.load(open('finalized_nb.sav', 'rb'))

inputs = []
emotion = []
response1 = []
response2 = []

#Read in the input comparison file
with open('input_comparison.csv', encoding='UTF-8') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        inputs.append(row[0])
        emotion.append(row[1])
        response1.append(row[2])
        response2.append(row[3])

input_sentences = rework_text(inputs, True)




#Say hello to the user
print('Hi!')


#Until the user asks to stop the chatbot will talk to them
stop = False

while not stop:
    
    highest_sim = 0
    found = False
    text_list = []
    
    #First print the user's name, then ask them for an input
    user_input = input('Please enter a sentence, or to bring this conversation to an end ask me to stop:\n')
    

    #Lower and tokenize the user's input to make it workable
    user_input = user_input.lower()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    user_token = []
    user_token.append(tokenizer.tokenize(user_input))
    
    #As user_token makes a list of lists use this to just get
    #the list of words as tokens
    token_list = user_token[0]
    
    if token_list[0] == "stop" and len(token_list) == 1:
        print("Goodbye!")
        stop = True
    
    
    #checks for similarity in the dataset of inputs it has          
    if stop == False: 
        
        for i in range(len(input_sentences)):
            sim_value = get_similarity(token_list, input_sentences[i])
            if sim_value >= highest_sim:
                match = i
                highest_sim = sim_value
            
        #If a high enough match is found print the corresponding text
        if highest_sim > 0.5:
            found = True
            
            text_list.append(user_input)
            new_X = text_list

            ## NLU
            # this is where BERT needs to classify the text
            #in place of the nb classifier
            #new_y = nb_model.predict(new_X)
            new_y = 

            #This is here to guage accuracy of classifier prediction
            print("Predicted emotion:", new_y)
             
            #irrelevant to the final code
            #This was simply to show that a response can be selected from a file if needed
            #choice = random.randint(1,2)
            #if choice ==1:
            #    print(response1[match], "\nExpected emotion:", emotion[match])
            #else:
            #    print(response2[match], "\nExpected emotion:", emotion[match])
     
            ## NLG
            # this is where the code to generate a response needs to go    
     
        
    #If no response can be matched to an input the chatbot informs a user
    #that they don't know how to responde 
    if stop == False and found == False:
        print("I'm sorry I dont understand, please say something else and I'll try again!")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    