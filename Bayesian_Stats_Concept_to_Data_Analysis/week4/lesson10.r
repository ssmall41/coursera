# Restaurant A
#n=30
#sample_mean=622.8
#sample_var=403.1

# Restaurant B
n=27
sample_mean=609.7
sample_var=401.8

# Prior hyperparameters
a=3
b=200
w=0.1
m=500

# How many draws should we do for the sampling?
num_draws=1000

# Calculate the posterior distribution
a_t = a + n/2
b_t = b + (n-1)/2*sample_var + (w*n)/(2*(w+n))*(sample_mean - m)^2
m_t = (n*sample_mean + w*m)/(w+n)

z=rgamma(num_draws, shape=a_t, rate=b_t)
sig2=1/z
mu=rnorm(num_draws, mean=m_t, sd=sqrt(sig2/(w+n)))

