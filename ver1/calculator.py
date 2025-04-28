from counter import counter_vehicle
count_list = []
tl = 4 #start-up lost time
total_lost_time = 4 #time taken for vehicles to start moving, typically from 4-5 seconds
initial_critical_lane_volume = 0.25 #sum of all highest lane volume per hour, assume to be 0.55

def density_calculation():
    count = counter_vehicle()
    new_critical_lane_volume = count / 3600  # lane volume after each phase (1hr)
    print("vehicle count = ",count)
    g_min = tl+2*count #minimum green light time allowed
    initial_cycle_length = total_lost_time/(1-initial_critical_lane_volume) #length of each cycle of loop
    g_effective = (initial_cycle_length-total_lost_time)*(initial_critical_lane_volume/new_critical_lane_volume) #the amount of time during which vehicles can actually move through an intersection, not just the duration of the green light itself
    g_max = 1.25*g_effective  #maximum green light time allowed
    print("minimum green time: ", g_min, "maximum green time: ", g_max)
def main():
    density_calculation()
    print("minimum green time allowed by police: 40 maximum green time allowed by police: 110")


if __name__ == "__main__":
    main()
