import time
import random
from statistics import mean
import pandas as pd


# this simulation has been validated and works
def simulation(number_machines, number_jobs, warm_up, func, due_date_tightness, utilization, random_seed=None, decision_situtation=None, missing_operation=None):

    # Initialize decision dict
    if decision_situtation==True:
        decision_dict = {"decision": [], "PT": [], "RT": [], "RPT": [], "RNO": [], "DD": [], "RTO": [], "PTN": [], "SL": [], "WT": [], "APTQ": [], "NJQ": [], "WINQ": [], "CT": []}
    # Initialize Lists
    schedule, results, jobs, jobs_var, jobs_finished = [], [], [], [], []

    # Initialize global clock
    global_clock = 0

    # Initialize random seed if applicable
    if random_seed != None:
        random.seed(random_seed)


    # Set number of operations equal to number of machines (full shop mode)
    #number_operations = number_machines
    # Calculate the inter-arrival time
    mean_processing_time = 25
    if missing_operation==True:
        mean_number_operations = (number_machines+2) / 2
    else:
        mean_number_operations = number_machines

    interarrival_time = (mean_processing_time*mean_number_operations)/(number_machines*utilization)
    #interarrival_time = mean_processing_time

    # Initialize global parameters
    SPT, TRNO = 0, 0

    class Job():
        def __init__(self):
            self.start = 0
            self.end = 0
            self.clock = 0
            self.operations = []
            self.number_operations = 0
            self.RPT = 0
            self.RNO = 0
            self.DD = 0
            self.operation_to_release = int(0)
            self.next_operation = int(1)
            self.release_status = 'no'
            self.t_event = 0
            self.number = 0
            self.release_time = 0

        class Operation():
            def __init__(self):
                self.number = 0
                self.start = 0
                self.end = 0
                self.clock = 0
                self.PT = 0
                self.machine = int(999999)
                self.release_time = 0

    class Machine():
        def __init__(self):
            self.queue = {'Job':[], 'Operation':[], 'Priority':[]}
            self.job_to_release = []
            self.num_in_system = 0
            self.clock = 0.0
            self.t_depart = float('inf')
            self.t_event = 0
            self.status = 'Idle'
            self.current_job_finish = 0
            self.counter = 0
            #self.TRNO = 0
            #self.SPT = 0

        def execute(self):
            # update priority
            self.update_priority()
            # select the waiting operation with the lowest priority value
            min_priority = min(self.queue["Priority"])
            index_job = self.queue["Priority"].index(min_priority)
            next_job = self.queue['Job'][index_job].number
            #print(f'next job to be processed: {next_job}')
            #print('\n')
            # update operation and job data
            self.queue['Operation'][index_job].start = self.clock
            self.queue['Operation'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].t_event = self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].clock += self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].RPT -= self.queue["Operation"][index_job].PT
            self.queue['Job'][index_job].RNO -= 1
            self.queue['Job'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
            if self.queue['Operation'][index_job].number == 0:
                self.queue['Job'][index_job].start = self.clock
            if self.queue['Operation'][index_job].number == (self.queue['Job'][index_job].number_operations-1):
                self.queue['Job'][index_job].end = self.clock + self.queue["Operation"][index_job].PT
                jobs_var.remove(self.queue['Job'][index_job])
                jobs_finished.append(self.queue['Job'][index_job])
            self.t_event = self.queue["Operation"][index_job].PT
            self.clock += self.t_event

            self.current_job_finish = self.clock

            # set job status to 'release'
            self.queue['Job'][index_job].operation_to_release += 1
            self.queue['Job'][index_job].next_operation += 1
            self.queue['Job'][index_job].release_status = 'yes'
            self.queue['Job'][index_job].clock = self.clock

            # remove operation from queue
            del self.queue["Job"][index_job]
            del self.queue["Operation"][index_job]
            del self.queue["Priority"][index_job]

            # set status to 'running'
            self.status = 'Running'

        def update_priority(self):
            PT_list = []
            for i in range(len(self.queue['Job'])):
                PT_list.append(self.queue['Operation'][i].PT)
            APTQ = mean(PT_list)
            NJQ = len(self.queue['Job'])

            try:
                decision_number = max(decision_dict['decision'])
            except:
                decision_number = 0
            decision_number += 1

            #print('New decision')

            for i in range(len(self.queue['Job'])):
                PT = self.queue['Operation'][i].PT
                RT = self.queue['Job'][i].release_time
                RPT = self.queue['Job'][i].RPT
                RNO = self.queue['Job'][i].RNO
                DD = self.queue['Job'][i].DD
                RTO = self.queue['Operation'][i].release_time
                CT = self.clock
                SL = DD-(CT+RPT)
                WT = max(0, CT-RTO)
                next_operation_1 = self.queue['Job'][i].next_operation
                if next_operation_1 >= len(self.queue['Job'][i].operations):
                    PTN = 0
                    WINQ = 0
                else:
                    next_operation_1 = self.queue['Job'][i].operations[next_operation_1]
                    PTN = next_operation_1.PT
                    machine_next_operation = next_operation_1.machine
                    queue_next_operation = machines[machine_next_operation].queue
                    WINQ = sum(queue_next_operation['Operation'][i].PT for i in range(len(queue_next_operation['Job']))) \
                           + max(machines[machine_next_operation].clock - CT, 0)

                expected_waiting_time = 0
                next_operation_2 = self.queue['Job'][i].next_operation
                while next_operation_2 < len(self.queue['Job'][i].operations):
                    next_operation = self.queue['Job'][i].operations[next_operation_2]
                    machine_next_operation = next_operation.machine
                    queue_next_operation = machines[machine_next_operation].queue
                    expected_waiting_time += (sum(queue_next_operation['Operation'][i].PT for i in range(len(queue_next_operation['Job']))) -
                            max(machines[machine_next_operation].clock - CT, 0)) / 2
                    next_operation_2 += 1



                if decision_situtation==True:
                    decision_dict['decision'].append(decision_number)
                    decision_dict["PT"].append(PT)
                    decision_dict["RT"].append(RT)
                    decision_dict["RPT"].append(RPT)
                    decision_dict["RNO"].append(RNO)
                    decision_dict["DD"].append(DD)
                    decision_dict["RTO"].append(RTO)
                    decision_dict["PTN"].append(PTN)
                    decision_dict["SL"].append(SL)
                    decision_dict["WT"].append(WT)
                    decision_dict["APTQ"].append(APTQ)
                    decision_dict["NJQ"].append(NJQ)
                    decision_dict["WINQ"].append(WINQ)
                    decision_dict["CT"].append(CT)
                    priority = PT
                    self.queue["Priority"][i] = priority
                else:
                    #priority = func(PT, RT, RPT, RNO, DD, RTO, PTN, SL, WT, APTQ, NJQ, WINQ, CT)
                    priority = (2*PT + WINQ + PTN)
                    #priority = WINQ+PT
                    #priority = WINQ
                    #priority = PT
                    priority = DD

                    slack = DD - CT - RPT
                    sto = slack/RNO
                    priority = sto

                    if slack < 0:
                        priority = 1
                    elif slack >= expected_waiting_time:
                        priority = 0
                    else:
                        priority = (expected_waiting_time-slack) / expected_waiting_time

                    priority = priority/PT

                    self.queue["Priority"][i] = -priority

                #print(f'Priority: {priority}')
                #print('\n')



    class JobGenerator():
        def __init__(self):
            self.clock = 0.0
            self.number = 1

        def execute(self):
            # generate job
            job = Job()
            job.release_time = self.clock
            allowed_values = list(range(0, number_machines))
            total_processing_time = 0
            if missing_operation == True:
                job.number_operations = random.randint(2,number_machines)
            else:
                job.number_operations = number_machines
            #job.number_operations = 10
            number_operations = job.number_operations
            job.operations = [job.Operation() for o in range(job.number_operations)]
            for o in job.operations:
                o.PT = random.randint(1, 49)
                o.machine = random.choice(allowed_values)
                total_processing_time += o.PT
                o.number = job.operations.index(o)
                allowed_values.remove(o.machine)
            #print(number_operations)
            #print(total_processing_time)
            #print(job.release_time)
            job.DD = job.release_time + (due_date_tightness * total_processing_time)
            #print(job.DD)
            job.RPT = total_processing_time
            job.RNO = len(job.operations)
            job.number = self.number
            jobs.append(job)
            jobs_var.append(job)

            number_of_released_operation = job.operation_to_release
            machine_to_release = job.operations[number_of_released_operation].machine
            machines[machine_to_release].queue['Job'].append(job)
            machines[machine_to_release].queue['Operation'].append(job.operations[number_of_released_operation])
            machines[machine_to_release].queue['Priority'].append(0)
            job.operations[number_of_released_operation].release_time = self.clock
            interarrival_time_current = random.expovariate(1/interarrival_time)
            #print(interarrival_time_current)
            self.clock += interarrival_time_current
            self.number +=1

            return total_processing_time, number_operations

    # generate machines
    machines = [Machine() for _ in range(number_machines)]


    # generate Job generator
    job_generator = JobGenerator()

    # execute Job generator the first time to generate a first job
    processing_time, number_operations = job_generator.execute()
    # update global parameters
    TRNO += number_operations
    SPT += processing_time

    # start simulation
    # loop until stopping criterion is met
    while len(jobs_finished) < number_jobs:
        #print(len(jobs_finished))
        # check if there are operations to be released on each job
        for j in jobs_var:
            if j.clock <= global_clock:
                if j.release_status == 'yes':
                    number_of_released_operation = j.operation_to_release
                    if number_of_released_operation <= (len(j.operations)-1):
                        machine_to_release = j.operations[number_of_released_operation].machine
                        machines[machine_to_release].queue['Job'].append(j)
                        machines[machine_to_release].queue['Operation'].append(j.operations[number_of_released_operation])
                        machines[machine_to_release].queue['Priority'].append(0)
                        j.release_status = 'no'
                        j.operations[number_of_released_operation].release_time = j.end

        # check if there is a job to be released on the job generator
        if job_generator.clock <= global_clock:
            processing_time, number_operations = job_generator.execute()
            # update global parameters
            TRNO += number_operations
            SPT += processing_time


        # check if there are jobs waiting in the queue on each machine
        for i in machines:
            if i.clock <= global_clock:
                if len(i.queue["Job"]) != 0:
                    i.execute()
                    # update global parameters
                    TRNO -= 1
                    SPT -= i.t_event


        # check for next event on the three classes (jobs, machines, jobgenerator)
        t_next_event_list = []
        for m in machines:
            if m.clock > global_clock:
                t_next_event_list.append(m.clock)
        for j in jobs_var:
            if j.clock > global_clock:
                t_next_event_list.append(j.clock)
        if job_generator.clock > global_clock:
            t_next_event_list.append(job_generator.clock)

        # next event and update of global clock
        if t_next_event_list != []:
            t_next_event = min(t_next_event_list)
        else:
            t_next_event=0
        global_clock=t_next_event

        # set the machine times to the global time for those that are less than the global time
        for i in machines:
            if i.clock <= global_clock:
                i.clock = global_clock
        for j in jobs_var:
            if j.clock <= global_clock:
                j.clock = global_clock


    # Postprocessing

    # create schedule
    schedule = pd.DataFrame(results)

    # calculate performance measures
    #makespan = np.max(schedule['Finish'])
    max_tardiness = max([max((j.end - j.DD), 0) for j in jobs_finished[warm_up:]])
    #waiting_time = np.sum((j.end-j.start) for j in jobs_finished[warm_up:])
    mean_flowtime = mean([(j.end - j.release_time) for j in jobs_finished[warm_up:]])
    #mean_tardiness = mean([max((j.end - j.DD), 0) for j in jobs_finished[warm_up:] if max((j.end - j.DD), 0) > 0])
    mean_tardiness = mean([max((j.end - j.DD), 0) for j in jobs_finished[warm_up:]])
    if decision_situtation==True:
        return decision_dict
    else:
        return mean_flowtime, mean_tardiness, max_tardiness


#Test the algorithm
start = time.time()
max_tardiness = []
mean_tardiness = []
mean_flowtime = []
#random_seed = [2, 5, 13, 244, 42, 36, 53, 62, 123, 245, 56, 21, 89, 143, 201, 173, 73, 19, 321, 4]
#random_seed = [4, 15, 384, 199, 260]
random_seed = [255, 49, 201, 126, 50, 133, 118, 13, 249, 93, 60, 82, 221, 23, 196, 45, 157, 5, 171, 298, 122, 67, 280, 132, 138, 142, 38, 4, 199, 279, 80, 79, 273, 145, 274, 216, 83, 98, 193, 278, 155, 227, 258, 56, 43, 48, 73, 81, 63, 29]
# test it for 20 replications
random_seed = [255, 49, 201, 126, 50, 133, 118, 13, 249, 93, 60, 82, 221, 23, 196, 45, 157, 5, 171, 298]
for i in random_seed:
    current_mean_flowtime, current_mean_tardiness, current_max_tardiness = \
        simulation(number_machines=10, number_jobs=2500, warm_up = 500,  func=None, random_seed=i, due_date_tightness=4, utilization=0.80, missing_operation=True)
    max_tardiness.append(current_max_tardiness)
    mean_tardiness.append(current_mean_tardiness)
    mean_flowtime.append(current_mean_flowtime)
end = time.time()

#schedule.to_excel('schedule.xlsx')
print(mean_flowtime)
print(mean_tardiness)
print(max_tardiness)


print(f'Execution time simulation per replication: {(end - start)}')
print(f'Mean flowtime: {mean(mean_flowtime)}')
print(f'Mean Tardiness: {mean(mean_tardiness)}')
print(f'Max tardiness: {mean(max_tardiness)}')
#print(f'Mean tardiness: {mean(mean_tardiness)}')







