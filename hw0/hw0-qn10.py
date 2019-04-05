
# coding: utf-8

# In[14]:


import numpy as npµ
import matplotlib.pyplot as plt


# In[31]:


mean = [0, 0]
cov = [[1, 0], [0, 1]]  # identity
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis([-5,5,-5,5])
plt.savefig("Desktop/csm146/hw0/10a.pdf")


# In[32]:


mean = [1, 1]
cov = [[1, 0], [0, 1]]  # identity
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis([-5,5,-5,5])
plt.savefig("Desktop/csm146/hw0/10b.pdf")


# In[36]:


mean = [0, 0]
cov = [[2, 0], [0, 2]]  # identity
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis([-5,5,-5,5])
plt.savefig("Desktop/csm146/hw0/10c.pdf")


# In[34]:


mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # identity
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis([-5,5,-5,5])
plt.savefig("Desktop/csm146/hw0/10d.pdf")


# In[37]:


mean = [0, 0]
cov = [[1, -0.5], [-0.5, 1]]  # identity
x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.axis([-5,5,-5,5])
plt.savefig("Desktop/csm146/hw0/10e.pdf")


# In[42]:


from numpy import linalg as LA
a = [[1,0], [1,3]]
#w; v = LA.eig(a)
w, v = LA.eig(np.array([[1,0], [1,3]]))


# In[45]:


w


# In[46]:


v

