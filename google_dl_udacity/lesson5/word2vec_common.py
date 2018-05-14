import zipfile
import collections
import tensorflow as tf

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_dataset(words, vocabulary_size): #the input words is a list of words from wiki text, lots of duplicates
  count = [['UNK', -1]] #list
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) #example output: [('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine', 250430), ('two', 192644)]
  #count looks like: [['UNK', -1], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201)]
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  #dictionary looks like this: {'UNK': 0, 'the': 1, 'of': 2, 'and': 3, 'one': 4, 'in': 5}, basically ordered list starting from the most common words
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  #data is basically the same as words but a list of index instead of the string itself
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  #this is used for looking up the word string from its index
  return data, count, dictionary, reverse_dictionary