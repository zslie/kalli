import json
from spellchecker import SpellChecker

# takes in Response JSON from AWS
def check(awsResponseItem):
    blocks = awsResponseItem["Blocks"]
    if (blocks is None):
        print("Blocks is None")
        return
    lines = [obj for obj in blocks if(obj['BlockType'] == "LINE")] 
    textLines = [item.get('Text') for item in lines]
    splits = [item.split(" ") for item in textLines]
    print(splits)
    
def extractLines(awsResponseItem):
    print("nice.")

def updateChild(id, newSpelling, awsResponseItem):
    print("nice.")