# coding=utf8
"""
Active Learning on Graphs via Generalization Error Bound Minimization
"""
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.linalg import block_diag
from numpy import linalg as LA

class ActiveLearning():
    '''
    Input: A = Adjacency matrix,
           l = number of nodes to select
           μ = regularization parameter
           e = small positive value to substitute the zero eigenvalues
           l = number of ndes to select
           n = order of each complete graph
    '''

    def __init__(self, m, l, n ,e):
        self.m = m
        self.l = l
        self.n = n
        self.e = e
        self.A = self.create_initial_adj_matrix()
        self.H = None
        self.H_k = None


    #active learning algorithm: generalization error bound minimization
    def compute_generalization_error_bound_minimization(self):
        L = self.compute_normalized_laplacian_matrix(self.A)
        w, v = self.compute_eigen_decomposition(L)
        most_informative_nodes = self.compute_most_informative_nodes(w, v)
        return most_informative_nodes

    #computes the set of most informative nodes
    def compute_most_informative_nodes(self, w, v):
        self.H = self.compute_initial_H_matrix(w)
        print "initial H matrix", self.H
        #initial empty set
        most_informative_nodes = set()
        #'''
        for i in range(0, self.l):
            most_informative_node = self.compute_most_informative_node(w, v, i, most_informative_nodes)
            most_informative_nodes.add(most_informative_node)
            print "most info nodes at iteration ",i,":", most_informative_nodes
            #update H
            self.update_H_matrix(v, most_informative_node)
        #'''
        print len(self.A)
        return most_informative_nodes

    #update H matrix
    def update_H_matrix(self, v, index):
        u_i = np.array(v[index, :])
        u_i = u_i.T
        temp = np.dot(self.H_k, u_i)
        numerator = np.dot(self.H_k, u_i).T
        numerator = np.dot(temp, numerator)
        #print "numerator ", numerator
        denominator = np.dot(u_i.T, self.H_k)
        denominator = 1 + np.dot(denominator, u_i)
        #print "denominator ", denominator
        self.H_k = self.H_k - (numerator / denominator)

    #compute the most informative node from the seed pool
    def compute_most_informative_node(self, w,v, index, selected):
        max_score = -1
        max_index = -1
        for i in range (0, len(self.A)):
            u_i = np.array(v[i,:])
            u_i = u_i.T
            #print u_i
            if index == 0:
                self.H_k = np.array((1/self.H))
                self.H_k[self.H_k == np.inf] = 0

            temp = np.dot( self.H_k,  u_i)
            D = np.diag(w)
            D = 1/D
            D[D == np.inf] = 0
            #print "D ",D
            numerator = np.dot(temp.T, D)
            numerator = np.dot(numerator, temp)
            #print "numerator ", numerator
            denominator = np.dot(u_i.T, self.H_k )
            denominator = 1 + np.dot(denominator, u_i)
            #print "denominator ", denominator
            score = numerator/denominator
            #print "score ==> ", score
            if score > max_score and i not in selected:
                max_score = score
                max_index = i
        return max_index

    #create initial adjacency matrix from corpus frequency list
    def create_initial_adj_matrix(self):
        G = nx.complete_graph(self.n)
        M= np.matrix(block_diag(nx.to_numpy_matrix(G), nx.to_numpy_matrix(G), nx.to_numpy_matrix(G)))
        #print "M, ",M
        return M

    #computes the normalized laplacian matrix
    def compute_normalized_laplacian_matrix(self, M):
        G = nx.from_numpy_matrix(M)
        L = nx.normalized_laplacian_matrix(G)
        #print "L ",L
        return L

    #performs eigen decomposition
    def compute_eigen_decomposition(self, L):
        w, v = LA.eig(nx.to_numpy_matrix(nx.Graph(L)))
        #print "w ", w
        #print "v ", v
        return w,v

    #computes H0
    def compute_initial_H_matrix(self, w):
        #print "w, ", w
        #w[w  == 0] += self.e
        print "w, ", w
        h = 1 / ((self.m * w + 1) * (self.m * w + 1) - 1)
        h[h == np.inf] = 0
        h[np.isnan(h)] = 0
        #print "h, ",h
        H_0 = np.diag(h)
        #print "H_0 ", H_0
        return H_0

if __name__ == '__main__':
    #m = 0.01 --> regularization parameter
    #l = 5 --> number of nodes to select
    #n = 30 --> order of each complete graph
    # e =  small positive value to substituve the zero eigen values

    al = ActiveLearning(0.01, 5, 30, 0.000001)
    #al.create_initial_matrix(9)
    al.compute_generalization_error_bound_minimization()