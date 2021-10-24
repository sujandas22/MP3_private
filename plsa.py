import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        print("Inside the build_corpus >>>>>")
        documentsCount = 0
        documents = self.documents
		
        with open(self.documents_path) as file:
            for documents in file.readlines():
                    documents = documents.rstrip('}\n ').strip('0\t').strip('1\t').split(' ')
                    documentsCount = documentsCount +1
                    self.documents.append(documents)
			
            self.number_of_documents = documentsCount

    
		
    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        print("initialize_randomly...")
        self.document_topic_prob = normalize(np.random.random_sample((self.number_of_documents, number_of_topics)))
        self.topic_word_prob = normalize(np.random.random_sample((number_of_topics, len(self.vocabulary))))
		
		 
    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """

        print("Inside build_term_doc_matrix  >>> ")
        self.term_doc_matrix = np.zeros([self.number_of_documents,self.vocabulary_size])
        for kVal in range(0, self.number_of_documents):
            for lVal,wordVocab in enumerate(self.vocabulary):
                wrd_doc = 0
                for oVal in range(0, len(self.documents[kVal])):
                     if (wordVocab == self.documents[kVal][oVal]):
                         wrd_doc = wrd_doc +1
                self.term_doc_matrix[kVal][lVal] = wrd_doc
        #print("term_doc_matrix >>> " + self.term_doc_matrix)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        
        for iCount in range(0,len(self.documents)):
            for jCount in range(iCount,len(self.documents[iCount])):
                self.vocabulary.append(self.documents[iCount][jCount])

        self.vocabulary = set(self.vocabulary)
		
        self.vocabulary = sorted(self.vocabulary)
		#print("Value of the vocabulary")
        self.vocabulary_size = len(self.vocabulary)
		
		 
    

        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self,number_of_topics):
        """ The E-step updates P(z | w, d)
        """
        createdDataMat = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)
        print("E step:")
		#print("Initial value of createdDataMat >>>>>"+createdDataMat)
        
  	
        for docValue in range(0, self.number_of_documents):
            for wordValue in range(0, self.vocabulary_size):
                recordSum = 0
                for topicValue in range(0, number_of_topics):
                    self.topic_prob[docValue][topicValue][wordValue] = (self.document_topic_prob[docValue][topicValue] * self.topic_word_prob[topicValue][wordValue])
                    recordSum = recordSum + self.topic_prob[docValue][topicValue][wordValue]
                for topicValue in range(0, number_of_topics):
                    self.topic_prob[docValue][topicValue][wordValue] = self.topic_prob[docValue][topicValue][wordValue]/recordSum

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        #print("Maximization_step has started >>>>>"+number_of_topics)
        
    
		 
        for topicVal in range(0, number_of_topics):
            for wordVal in range(0, self.vocabulary_size):
                createdDataMat = 0
                for docVal in range(0,self.number_of_documents):
                    createdDataMat = createdDataMat+ (self.term_doc_matrix[docVal][wordVal]*self.topic_prob[docVal][topicVal][wordVal])
                self.topic_word_prob[topicVal][wordVal] = createdDataMat
        self.topic_word_prob = normalize(self.topic_word_prob)
        for docVal in range(0,self.number_of_documents):
             for topicVal in range(0,number_of_topics):
                self.document_topic_prob[docVal][topicVal] = 0
                for wordVal in range(0, self.vocabulary_size):
                    self.document_topic_prob[docVal][topicVal] = self.document_topic_prob[docVal][topicVal]  +   (self.term_doc_matrix[docVal][wordVal]*self.topic_prob[docVal][topicVal][wordVal])
        self.document_topic_prob = normalize(self.document_topic_prob)
        #print("Document Topic Problem >>> "+self.document_topic_prob)



    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        print("Start of calculate_likelihood...")
        maxNumberOfWrd = 0
        for docValue in range(0,self.number_of_documents):
            for wordValue in range(0,self.vocabulary_size):
                maxTopicVal = 0
                for topicValue in range(0,number_of_topics):
                    maxTopicVal = maxTopicVal +  ((self.document_topic_prob[docValue][topicValue]*self.topic_word_prob[topicValue][wordValue]))
                if maxTopicVal > 0:
                    maxNumberOfWrd =  maxNumberOfWrd + (np.log(maxTopicVal) * self.term_doc_matrix[docValue][wordValue])
		#print("Value of maxNumberOfWrd >>> "+maxNumberOfWrd)
        self.likelihoods.append(maxNumberOfWrd)
		#print("End of calculate_likelihood...")
        return

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            
            self.expectation_step(number_of_topics)
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            if (iteration >1 and (self.likelihoods[iteration] - self.likelihoods[iteration-1] < epsilon) ):
                break
        print(self.likelihoods)






def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Size of the Vocabulary present > " + str(len(corpus.vocabulary)))
    print("Length of the num of docs > " + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
