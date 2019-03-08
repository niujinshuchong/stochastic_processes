import numpy as np
import random
import math
import matplotlib.pyplot as plt

arrival_rate = 2
arrival_service_ratio = 0.8
service_rate = arrival_rate / arrival_service_ratio
print(1. / ( service_rate - arrival_rate), arrival_rate / (service_rate * (arrival_rate - service_rate)))
# simulation
trial = 100000

# since arrival time and and service time are exponential distribution,
# we first generate all arrival time and service time with exponential distribution
# then we do the simulation: check every ones response time, queueing time

arrival_times = np.random.exponential(1. / arrival_rate, trial)
service_times = np.random.exponential(1. / service_rate, trial)

arrival_times = np.cumsum(arrival_times)
response_times = np.zeros_like(arrival_times)
queueing_times = np.zeros_like(arrival_times)
leave_times = np.zeros_like(arrival_times)
end_of_last_service = 0.0

# service for every one
for i in range(trial):
    # no body is waiting
    if arrival_times[i] >= end_of_last_service:
        queueing_times[i] = 0
        response_times[i] = service_times[i]
        end_of_last_service = arrival_times[i] + service_times[i]
        leave_times[i] = end_of_last_service
    # some one is waiting
    else:
        queueing_times[i] = end_of_last_service - arrival_times[i]
        response_times[i] = queueing_times[i] + service_times[i]
        end_of_last_service += service_times[i]
        leave_times[i] = end_of_last_service

# simulation ends when last person arrivals
leave_times = leave_times[leave_times < arrival_times[-1]]
# number of jobs in the system
arrival_count = np.ones_like(arrival_times)
leave_count = - np.ones_like(leave_times)
count = np.concatenate((arrival_count, leave_count), axis=0)
times = np.concatenate((arrival_times, leave_times), axis=0)

count = count[times.argsort(axis=0)]
times = times[times.argsort(axis=0)]
count = np.cumsum(count)

print('the mean and variance of the number of jobs in the system')
mean = np.sum((times[1:] - times[:-1]) * count[:-1]) / arrival_times[-1]
var = np.sum((times[1:] - times[:-1]) * (count[:-1] - mean) ** 2 )  / arrival_times[-1]
print(mean, var)

print('the mean response time of jobs in the system')
print(np.mean(response_times))
print('the mean queueing time of jobs in the system')
print(np.mean(queueing_times))

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(times, count)
plt.ylabel('jobs in system')

plt.subplot(234)
plt.hist(queueing_times, density=True)
plt.title('distribution of queueing time')
plt.ylabel('density')
plt.xlabel('time')

plt.subplot(235)
plt.hist(response_times, density=True)
plt.title('distribution of response time')
plt.ylabel('density')
plt.xlabel('time')

plt.subplot(236)
plt.hist(count, density=True)
plt.title('distribution of jobs')
plt.ylabel('density')
plt.xlabel('number of jobs in system')

plt.savefig("mm1_queue_%.1lf.png"%(arrival_service_ratio), dpi=300)
plt.show()
