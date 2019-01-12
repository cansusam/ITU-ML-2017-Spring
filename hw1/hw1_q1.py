# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

# q1-a

p1 = 0.5
p2 = 0.5
mu1 = -5
mu2 = -10
variance = 5
sigma = math.sqrt(variance)

size = 10000

w = np.linspace(-20,5,size)

fig = plt.figure(figsize=(5,3))
ax1 = fig.add_axes([0.2, 0.1, 0.7, 0.75])

# instead of formulation, built-in normal distribution function is used
#p1_dist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (w - mu1)**2 / (2 * sigma**2) );
#p2_dist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (w - mu2)**2 / (2 * sigma**2) );
px_1 = mlab.normpdf(w, mu1, sigma);
px_2 = mlab.normpdf(w, mu2, sigma);

ax1.plot(w,px_1)
ax1.plot(w,px_2)
fig.suptitle("Q1-a PDF of two classes")
ax1.set_xlabel("x")
ax1.set_ylabel("P(x|Ci)")
fig.show()
fig.savefig("q1a1")
#plt.show()

# separation surface

fig1 = plt.figure(figsize=(5,3))
ax2 = fig1.add_axes([0.2, 0.1, 0.7, 0.75])

px = px_1*p1+px_2*p2;
p1_r = px_1*p1/px;
p2_r = px_2*p2/px;

ax2.plot(w,p1_r);
ax2.plot(w,p2_r);

sep = np.linspace(0,1,size);
ax2.plot(np.full(size,-7.5),sep)
fig1.suptitle("Q1-a Separating Surface")
ax2.set_xlabel("x")
ax2.set_ylabel("P(Ci|x)")
fig1.show()
fig1.savefig("q1a2")
#plt.show()

# q1-b

p1 = 0.8
p2 = 0.2

fig2 = plt.figure(figsize=(5,3))
ax3 = fig2.add_axes([0.2, 0.1, 0.7, 0.75])

px2 = px_1 * p1 + px_2 * p2;

p1_r2 = px_1 * p1 / px2;
p2_r2 = px_2 * p2 / px2;

ax3.plot(w, p1_r2);
ax3.plot(w, p2_r2);

# -8.886 : obtained after solving equation

ax3.plot(np.full(size, -8.886), np.linspace(0, 1, size));
fig2.suptitle("Q1-b Separating Surface with p1=0.2")
ax3.set_xlabel("x")
ax3.set_ylabel("P(Ci|x)")
fig2.show()
fig2.savefig("q1b")
#plt.show()

# q1-c-a

p1 = 0.5
p2 = 0.5
binSize = 100;

fig3 = plt.figure(figsize=(5,3))
ax4 = fig3.add_axes([0.2, 0.1, 0.7, 0.75])

# data set (s1 + s2) is created with equal probability
s1 = np.random.normal(mu1, sigma, (int)(size*p1))
s2 = np.random.normal(mu2, sigma, (int)(size*p2))

count1, bins1, ignored1 = plt.hist(s1, binSize, normed=True)
count2, bins2, ignored2 = plt.hist(s2, binSize, normed=True)

bins = np.concatenate((bins1, bins2), axis=0);
bins.sort()

p1_dist = mlab.normpdf(bins, mu1, sigma);
p2_dist = mlab.normpdf(bins, mu2, sigma);

ax4.plot(bins, p1_dist,linewidth=2, color='b')
ax4.plot(bins, p2_dist,linewidth=2, color='r')
fig3.suptitle("Q1-c-a Random dataset histogram")
ax4.set_xlabel("x")
ax4.set_ylabel("P(x|Ci)")
fig3.show()
fig3.savefig("q1ca1")
#plt.show()

# means and variances
print("\nq1-c-a")
print('P1 mean %d' % np.mean(s1))
print('P2 mean %d' % np.mean(s2))
print('P1 var %d' % np.var(s1))
print('P2 var %d' % np.var(s2))

with open('q1-c-a-results.txt', 'w') as the_file:
    the_file.write('P1 mean %f.2\n'% np.mean(s1))
    the_file.write('P2 mean %f.2\n'% np.mean(s2))
    the_file.write('P1 var  %f.2\n'% np.var(s1))
    the_file.write('P1 var  %f.2\n'% np.var(s2))

# separation surface

fig4 = plt.figure(figsize=(5,3))
ax5 = fig4.add_axes([0.2, 0.1, 0.7, 0.75])

px_3 = p1_dist*p1+p2_dist*p2;
p1_r = p1_dist*p1/px_3;
p2_r = p2_dist*p2/px_3;

ax5.plot(bins,p1_r);
ax5.plot(bins,p2_r);

# -7.5 : obtained after solving equation

ax5.plot(np.full(size,-7.5),np.linspace(0,1,size));
fig4.suptitle("Q1-c-a Separating surface for random dataset")
ax5.set_xlabel("x")
ax5.set_ylabel("P(Ci|x)")
fig4.show()
fig4.savefig("q1ca2")
#plt.show()

# q1-c-b

p1 = 0.8
p2 = 0.2
binSize = 100;

fig5 = plt.figure(figsize=(5,3))
ax6 = fig5.add_axes([0.2, 0.1, 0.7, 0.75])

# data set (s1 + s2) is created with equal probability
s1 = np.random.normal(mu1, sigma, (int)(size*p1))
s2 = np.random.normal(mu2, sigma, (int)(size*p2))

count1, bins1, ignored1 = plt.hist(s1, binSize, normed=True)
count2, bins2, ignored2 = plt.hist(s2, binSize, normed=True)

bins = np.concatenate((bins1, bins2), axis=0);
bins.sort()


p1_dist = mlab.normpdf(bins, mu1, sigma);
p2_dist = mlab.normpdf(bins, mu2, sigma);

ax6.plot(bins, p1_dist,linewidth=2, color='b')
ax6.plot(bins, p2_dist,linewidth=2, color='r')
fig5.suptitle("Q1-c-b Random dataset histogram for p1=0.8")
ax6.set_xlabel("x")
ax6.set_ylabel("P(x|Ci)")
fig5.show()
fig5.savefig("q1cb1")
#plt.show()

# means and variances
print("\nq1-c-b")
r1_mean = print('P1 mean %d' % np.mean(s1))
r2_mean = print('P2 mean %d' % np.mean(s2))
r1_var = print('P1 var %d' % np.var(s1))
r2_var = print('P2 var %d' % np.var(s2))

with open('q1-c-b-results.txt', 'w') as the_file:
    the_file.write('P1 mean %f.2\n'% np.mean(s1))
    the_file.write('P2 mean %f.2\n'% np.mean(s2))
    the_file.write('P1 var  %f.2\n'% np.var(s1))
    the_file.write('P1 var  %f.2\n'% np.var(s2))

fig6 = plt.figure(figsize=(5,3))
ax7 = fig6.add_axes([0.2, 0.1, 0.7, 0.75])

px_3 = p1_dist*p1+p2_dist*p2;
p1_r = p1_dist*p1/px_3;
p2_r = p2_dist*p2/px_3;

ax7.plot(bins,p1_r);
ax7.plot(bins,p2_r);

# -8.886 : obtained after solving equation

ax7.plot(np.full(size,-8.886),np.linspace(0,1,size));
fig6.suptitle("Q1-c-b Separating surface for random dataset p1=0.8")
ax7.set_xlabel("x")
ax7.set_ylabel("P(Ci|x)")
fig6.show()
fig6.savefig("q1cb2")
plt.show()