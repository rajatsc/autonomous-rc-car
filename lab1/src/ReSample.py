import rospy
import numpy as np

class ReSampler:
  def __init__(self, particles, weights, state_lock=None):
    self.particles = particles
    self.weights = weights
    self.particle_indices = None  
    self.step_array = None
    self.res_count = 0
    self.list_of_candidates = None
    self.new_num_particles=None
    self.change_num_particles=False
    
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
    

  def calculate_new_num_particles(self):
    
    if self.change_num_particles==True: 
        #Some code
      pass
    else:
      self.new_num_particles=len(self.particles)
  
  def resample_naiive(self):
   
    #self.calculate_new_num_particles()
    number_of_items_to_pick=self.new_num_particles  
    
    if self.list_of_candidates is None: 
      self.list_of_candidates=np.arange(len(self.particles))
    probability_distribution=self.weights  ##Enter normalized particle weights
    self.state_lock.acquire()
    draw = np.random.choice(self.list_of_candidates, self.particles.shape[0],replace=True, p=probability_distribution)
    new_particles=self.particles[draw]
    self.particles[:] = new_particles
    self.weights[:] = 1.0 / self.weights.shape[0]
    self.state_lock.release()
    self.res_count += 1
    print "resample: " + str(self.res_count)
    
  def resample_low_variance(self):
    self.state_lock.acquire()
    # Implement low variance re-sampling
    # YOUR CODE HERE
    self.calculate_new_num_particles()
    number_of_items_to_pick=self.new_num_particles
    
    r=np.random.uniform(low=0, high=1.0/(number_of_items_to_pick), size=1)

    new_particles=[]
    i=0
    c=self.weights[0]
    for m in range(number_of_items_to_pick):
      u=(r+(1.0/number_of_items_to_pick)*m)

      while (u > c):
        i=i+1
        c=c+self.weights[i]
      new_particles.append(self.particles[i])
    self.particles[:] = np.asarray(new_particles)
    self.weights[:] = 1.0 / self.weights.shape[0]

    self.state_lock.release()

