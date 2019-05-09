import numpy as np
import graphics
import rover
import math

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps
    
    # Compute the forward messages
    forward_prev = rover.Distribution()
    for i, observation in enumerate(observations):
        forward_curr = rover.Distribution()
        for current_state in all_possible_hidden_states:
            if i == 0:
                previous_f_sum = prior_distribution[current_state]
            else:
                previous_f_sum = sum(forward_prev[previous_state] * transition_model(previous_state)[current_state] \
                                     for previous_state in all_possible_hidden_states)
            
            if not observation:
                forward_curr[current_state] = previous_f_sum
            else:
                forward_curr[current_state] = observation_model(current_state)[observation] * previous_f_sum
                
        forward_curr.renormalize()
        forward_messages[i] = forward_curr
        forward_prev = forward_curr
 

    # Compute the backward messages
    backward_prev = rover.Distribution()
    for i, observation_i_plus in enumerate(reversed(observations[1:]+[(None,)])):
        backward_curr = rover.Distribution()
        for current_state in all_possible_hidden_states:
            if i == 0:
                backward_curr[current_state] = 1
            else:
                if not observation_i_plus:
                    backward_curr[current_state] = sum(transition_model(current_state)[next_state] \
                                                      * backward_prev[next_state] for next_state in all_possible_hidden_states)
                else:
                    backward_curr[current_state] = sum(transition_model(current_state)[next_state] \
                                                      * observation_model(next_state)[observation_i_plus] \
                                                      * backward_prev[next_state] for next_state in all_possible_hidden_states)
        backward_curr.renormalize()
        backward_messages[len(backward_messages)-i-1] = backward_curr
        backward_prev = backward_curr

   
    # Compute the marginals
    for i in range(len(observations)):
        marginal = rover.Distribution()
        for state in all_possible_hidden_states:
            marginal[state] = forward_messages[i][state] * backward_messages[i][state]
        marginal.renormalize()
        marginals[i] = marginal

    return marginals

def Log(num):
    if num == 0:
        return -math.inf
    else:
        return math.log(num)

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    previous_message = rover.Distribution()
    path_map = [dict() for x in range(100)]
    for i, observation in enumerate(observations):
        current_message = rover.Distribution()

        for current_state in all_possible_hidden_states:
            if not observation:
                log_p_X_given_Z = 0
            else:
                log_p_X_given_Z = Log(observation_model(current_state)[observation])
                
            if i == 0:
                current_message[current_state] = log_p_X_given_Z + Log(prior_distribution[current_state])
            
            else:
                max_found = -math.inf
                max_prev_state = None
                for previous_state in all_possible_hidden_states:
                    transition_prob = Log(transition_model(previous_state)[current_state]) + previous_message[previous_state]
                    if transition_prob > max_found:
                        max_found = transition_prob
                        max_prev_state = previous_state
                path_map[i][current_state] = max_prev_state
                current_message[current_state] = log_p_X_given_Z + max_found
                                                 
        previous_message = current_message


    # Reconstruct the Path
    current_state = max(current_message, key=lambda key: current_message[key])
    estimated_hidden_states = [current_state]
    for i in reversed(range(1, 100)):
        current_state = path_map[i][current_state]
        estimated_hidden_states.insert(0, current_state)

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

    count = 0
    for i in range(len(hidden_states)):
        if hidden_states[i] == marginals[i].get_mode():
            count += 1
        else:
            print(i, hidden_states[i], marginals[i].get_mode())
    print("Marginal Num Correct", count)

    count = 0
    for i in range(len(hidden_states)):
        if hidden_states[i] == estimated_states[i]:
            count += 1
        else:
            print(i, hidden_states[i], estimated_states[i])
    print("Viterbi Num Correct", count)