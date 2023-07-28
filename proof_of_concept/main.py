import json
import spellcheck

# Call AWS Textract and get word locations
# textractResponses = []
# for i in folderItems:
#     textractResponses.append(awsutil.textract_parse(i))

# Send spell check words through some program/statistics on what is would likely bespelledCorrectly = []
f = open('../sample/aws-textract-1/analyzeDocResponse.json')
textractResponses = json.load(f)
spelledCorrectly = spellcheck.check(textractResponses)

# Letter segmentation
# letterSegments = segmentation.segment(spelledCorrectly)

# run letters through NN to build image files
# images = imageGenerator.generateFontFiles()

# from image files build font file
# ttf = font.generateFromImages(images)
