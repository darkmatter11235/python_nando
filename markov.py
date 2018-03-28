import argparse
import random
class MarkovChain():

    def __init__(self, inputFilePath, messageLimit=100, total_output_msgs=10):
        self.chain = {}
        self.contents = []
        self.inputFilePath = inputFilePath
        self.messageSize = messageLimit
        self.total_num_msgs = total_output_msgs

    def generate_chain(self):
        index = 1
        words = self.contents.split(' ')
        for word in words[index:]:
            key = words[index-1]
            if key in self.chain:
                self.chain[key].append(word)
            else:
                self.chain[key] = [word]
            index += 1

    def read_contents(self):

        with open(self.inputFilePath, 'r') as file:
            self.contents = file.read().replace('\n\n', ' ')

    def generate_message(self):

        start = random.choice(self.chain.keys())
        messageSize = 1
        message = start
        currentWord = start
        while messageSize < self.messageSize:
            if currentWord in self.chain:
                nextWord = random.choice(self.chain[currentWord])
                message = message + ' ' + nextWord
                currentWord = nextWord
                messageSize += 1
            else:
                message = message + '\n'
                return message

        return message

    def run(self):
        self.read_contents()
        self.generate_chain()
        msg_num = 1
        while msg_num < self.total_num_msgs:
            message = self.generate_message()
            print "%d: %s\n" %(msg_num, message)
            msg_num +=1

parser = argparse.ArgumentParser(description="Process input for the markovchain")
                                 
parser.add_argument('inputFile')
parser.add_argument('--ml', type=int, help='Length of output msg')
parser.add_argument('--nm', type=int, help='Number of output msgs')

args = parser.parse_args()

messageLength = 100
numMsgs = 10

if  args.ml :
    messageLength = args.ml

if  args.nm :
    numMsgs = args.nm

mc = MarkovChain(args.inputFile, messageLength, numMsgs)
mc.run()
