from collections import defaultdict
import pandas as pd
import json
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re

class UserDataReader:
    def __init__(self,folder_path,data_csv_path, min_range, max_range):
        self.path = folder_path
        self.userdata_json = self.read_all_userdata(self.path, min_range, max_range)
        self.df = pd.read_csv(data_csv_path)
        
        
    def load_file(self,input_file):  
        #input_file = path.join(path)
        with open(input_file) as f:
            file = f.read()
        return file
    
    def read_all_userdata(self,path, min_range, max_range):
        userdata = []
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file_name in files:
#             print(file_name)
            res = file_name.split('_')
            res = res[1].split('.')
            sample_num = int(res[0])
            
            if "userInput" in file_name and sample_num >= min_range and sample_num < max_range:
                userdata.append(self.load_file(path+file_name))
        
        userdata_json = []
        for userInput in userdata:
            userdata_json.append(json.loads(userInput))
        return userdata_json       
        
        
    # returns a dictionary : Key ="StateName" | Value = "SummaryText"
    def read_summaries_keyvalue(self):
        sample_summaries = []
        for userInput in self.userdata_json:
            sample_summary={}
            for user_summary in userInput['summaries']:
                summary = ""
                for sen_id in user_summary['sentence_ids']:
                    summary += self.df[self.df['sid'] == sen_id]['sentence'].to_numpy()[0] + " "
                
                sample_summary[user_summary['state_name']]=summary.strip()
            sample_summaries.append(sample_summary)
        return sample_summaries  
    
    
    # returns a list of all summaries
    def read_summaries_list(self):
        sample_summaries = []
        for userInput in self.userdata_json:
            sample_summary=[]
            for user_summary in userInput['summaries']:
                summary = ""
                for sen_id in user_summary['sentence_ids']:
                    summary += self.df[self.df['sid'] == sen_id]['sentence'].to_numpy()[0] + " "
                sample_summary.append(summary.strip())
            sample_summaries.append(sample_summary)
        return sample_summaries 
    
     # returns a dictionary : Key ="Intent_Name" | Value = "List of StateNames"
    def read_intent_doc(self):
        sample_summaries = []
        for userInput in self.userdata_json:
            intent = userInput['intent']
            sample_summary=defaultdict(list)
            for user_summary in userInput['summaries']:
                sample_summary[intent].append(user_summary['state_name'])
            sample_summaries.append(sample_summary)
        return sample_summaries 
        
     # returns a dictionary : Key ="Intent_Name" | Value = "List of StateNames"    
    def read_example_summaries_sudocu(self):
        sample_summaries = []
        for userInput in self.userdata_json:
            summaries = userInput['summaries']
            sample_summaries.append(summaries)
        return sample_summaries 
        
    # returns a list of unique intents
    def read_intents(self):
        intents = set()
        for userInput in self.userdata_json:
            intent = userInput['intent']
            intents.add(intent)
        return list(intents)
    
    # list of lists
    # Each list has list of all used keywords for the summarization task
    def read_usedkeywords_doc(self):
        sample_summaries = []
        for userInput in self.userdata_json:
            sample_summary={}
            used_keywords = set()
            for user_summary in userInput['summaries']:
                current_keywords = user_summary['used_keywords']
                for key_word in current_keywords:
                    used_keywords.add(key_word)
            sample_summaries.append(list(used_keywords))
        return sample_summaries       