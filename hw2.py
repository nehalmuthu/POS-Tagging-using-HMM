import json
import collections

train_file = 'data/train'
dev_file = 'data/dev'
test_file = 'data/test'


vocab_file = 'vocab.txt'
model_file = 'hmm.json'
threshold = 2
tagset=set()


"""Task 1: Vocabulary Creation"""

# Counting the occurrences of each word in the training data
word_counts = collections.defaultdict(int)
with open(train_file, 'r') as f:
  for line in f:
    line = line.strip().split('\t')
    if len(line) == 3:
      tagset.add(line[2])
      word_counts[line[1]] += 1

#vocabulary
vocab = {word: count for word, count in word_counts.items() if count >= threshold}
#counting <unk> words
unk_count = sum(count for word, count in word_counts.items() if count < threshold)


# Write the vocabulary to a file
k=0
with open(vocab_file, 'w') as f:
  f.write(f"<unk>\t 0 \t {unk_count}\n")
  for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
    if word in vocab:
      k=k+1
      f.write(f"{word}\t{k}\t{count}\n")

print(f"Threshold for <unk>: {threshold}")
print(f"Vocabulary size: {k+1}") # +1 for the <unk>
print(f"Unknown words <unk> count: {unk_count}")
print("\n")

""" Task 2: Model Learning """

def learn_hmm(train_file, vocab, model_file):
    transition_counts = collections.defaultdict(int)
    emission_counts = collections.defaultdict(int)
    state_counts = collections.defaultdict(int)
    start_state_counts = collections.defaultdict(int)
    end_state_counts = collections.defaultdict(int)
    prev_state = None

    # Counting the transitions and emissions in the training data    
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                word, state = line[1], line[2]
                state_counts[state] += 1
                if prev_state is None:
                    start_state_counts[state] += 1
                if prev_state is not None:
                    transition_counts[(prev_state, state)] += 1
                if word in vocab:
                    emission_counts[(state, word)] += 1
                else:
                    emission_counts[(state,'<unk>')] += 1
                prev_state = state
            else:
                end_state_counts[prev_state] += 1
                prev_state = None
    
    # Calculating the transition and emission probabilities
    transition = {}
    for (prev_state, state), count in transition_counts.items():
        transition[f"{prev_state},{state}"] = count / state_counts[prev_state]

    for state, count in start_state_counts.items():
        transition[f"STARTING,{state}"] = count / sum(start_state_counts.values())
    
    for state, count in end_state_counts.items():
        transition[f"{state},ENDING"] = count / state_counts[state]
    
    emission = {}
    for (state, word), count in emission_counts.items():
        emission[f"{state},{word}"] = count / state_counts[state]

    # reding the emission and transition into a file
    hmm = {"transition": transition, "emission": emission}
    with open("hmm.json", "w") as f:
        json.dump(hmm, f)

# Training the model
with open(vocab_file, 'r') as f:
    vocab = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in f}

learn_hmm(train_file, vocab, model_file)

with open(model_file, 'r') as f:
    model = json.load(f)
    transition_params = model['transition']
    emission_params = model['emission']

print(f"Number of transition parameters: {len(transition_params)}")
print(f"Number of emission parameters: {len(emission_params)}")

print("\n")

"""Task 3: Greedy Decoding with HMM"""

def greedy_decode(words, vocab, transition_params, emission_params):
    tags = []
    for i in range(len(words)):
        word = words[i]

        if word not in vocab:
            word = "<unk>"
        max_tag = None
        max_prob = -1
        if i==0:
          for tag in tagset:
            transition_prob=transition_params.get(f"STARTING,{tag}",1e-7)
            emission_prob = emission_params.get(f"{tag},{word}", 1e-7)
            prob = transition_prob * emission_prob
            if prob > max_prob:
              max_prob = prob
              max_tag = tag
        else:
          for tag in tagset:
            transition_prob=transition_params.get(f"{tags[i-1]},{tag}",1e-7)
            emission_prob = emission_params.get(f"{tag},{word}", 1e-7)
            prob = transition_prob * emission_prob
            if prob > max_prob:
              max_prob = prob
              max_tag = tag

        tags.append(max_tag)

    return tags

"""TASK 3 - Accuracy on Dev Data"""

print("Training on dev using Greedy ....")
words = []
gold_tags = []
with open(dev_file, 'r') as f:
  for i, line in enumerate(f):
    line = line.strip().split('\t')
    if len(line) == 3:
      word, gold_tag = line[1], line[2]
      words.append(word)
      gold_tags.append(gold_tag)
tags = greedy_decode(words, vocab, transition_params, emission_params)

correct = 0
total = 0

for j in range(len(words)):
  if tags[j] == gold_tags[j]:
    correct += 1
  total += 1
accuracy = correct / total
    
print(f"Greedy Decoding with HMM - Accuracy on dev data: {accuracy}")

"""TASK 3 - Predictions on Test Data"""
print("Predicting on test using Greedy ....")

words = []
gold_tags = []
with open(test_file, 'r') as f:
  for line in f:
    line = line.strip().split('\t')
    if len(line) == 2:
      word = line[1]
      words.append(word)

tags = greedy_decode(words, vocab, transition_params, emission_params)

greedyOutput = open("greedy.out", "w")
k=0
with open(test_file, 'r') as f:
  for line in f:
    line = line.strip().split('\t')
    if len(line) == 2:
      idx,word  = line[0], line[1]
      tag=tags[k]
      k=k+1
      greedyOutput.write(f"{idx}\t{word}\t{tag}\n")
    else:
      greedyOutput.write(f"{line[0]}\n")    
f.close()
greedyOutput.close()
print("Prediction on Test using Greedy saved to greedy.out")
print("\n")

"""Task 4: Viterbi Decoding with HMM """

def viterbi_decode(words, vocab, transition_params, emission_params):
    tags = []
    n = len(words)
    pi = {}
    bp = {}
    for i in range(n):
        word = words[i]
        if word not in vocab:
            word = "<unk>"
        for tag in tagset:
            if i == 0:
                pi[f"{i},{tag}"] = transition_params.get(f"STARTING,{tag}", 1e-6) * emission_params.get(f"{tag},{word}", 1e-6)

            else:
                max_prob = -1
                max_prev_tag = None
                for prev_tag in tagset:
                    prob = pi[f"{i-1},{prev_tag}"] * transition_params.get(f"{prev_tag},{tag}", 1e-6) * emission_params.get(f"{tag},{word}", 1e-6)
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_tag = prev_tag
                pi[f"{i},{tag}"] = max_prob
                bp[f"{i},{tag}"] = max_prev_tag

    max_prob = -1
    max_end_tag = None
    for tag in tagset:
        prob = pi[f"{n-1},{tag}"] * transition_params.get(f"{tag},STOPPING", 1e-6)
        if prob > max_prob:
            max_prob = prob
            max_end_tag = tag

    tags = [max_end_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, bp[f"{i},{tags[0]}"])
    return tags

"""TASK 4 - Accuracy on Dev Data"""
print("Training on dev using Viterbi  ....")

def evaluate_viterbi_decoding_dev(dev_file, vocab, transition_params, emission_params):
    correct = 0
    total = 0
    words = []
    gold_tags = []
    with open(dev_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if len(line) == 3:
                word, gold_tag = line[1], line[2]
                words.append(word)
                gold_tags.append(gold_tag)
            else:
                tags = viterbi_decode(words, vocab, transition_params, emission_params)
                for j in range(len(words)):
                  if tags[j] == gold_tags[j]:
                    correct += 1
                  total += 1
                words = []
                gold_tags = []
    accuracy = correct / total
    return accuracy


accuracy = evaluate_viterbi_decoding_dev(dev_file, vocab, transition_params, emission_params)
print(f"Viterbi Decoding with HMM - Accuracy on dev data: {accuracy}")

"""TASK 4 - Predictions on Test Data"""
print("Predicting on Test using Viterbi ...")

def evaluate_viterbi_decoding_test(test_file, vocab, transition_params, emission_params):
    words = []
    result=[]
    with open(test_file, 'r') as f:
        for line in f:
          line = line.strip().split('\t')
          if len(line) == 2:
            word = line[1]
            words.append(word)
          else:
            tags = viterbi_decode(words, vocab, transition_params, emission_params)
            for t in tags:
              result.append(t)
            words = []
    tags = viterbi_decode(words, vocab, transition_params, emission_params)
    for t in tags:
      result.append(t)
    f.close()    
    return result            


tags = evaluate_viterbi_decoding_test(test_file, vocab, transition_params, emission_params)
viterbiOutput = open("viterbi.out", "w")
k=0
with open(test_file, 'r') as f:
  for line in f:
    line = line.strip().split('\t')
    if len(line) == 2:
      idx,word  = line[0], line[1]
      tag=tags[k]
      k=k+1
      viterbiOutput.write(f"{idx}\t{word}\t{tag}\n")
    else:
      viterbiOutput.write(f"{line[0]}\n")    
f.close()
viterbiOutput.close()
print("Prediction on Test using Viterbi saved to viterbi.out")