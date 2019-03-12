import numpy as np 
import pandas as pd 
import time

N_STATES = 10
ACTIONS = ['left', 'right']
ALPHA = 0.1 
GAMMA = 0.9 
EPSILON = 0.9 
MAX_EPISODES = 20
FRESH_TIME = 0.3  
def Q_Table(n_states,actions):
	table = pd.DataFrame(np.ones((n_states,len(actions))),columns = actions)
	print(table)
	return table
	

def choose_action(s,q_table):
	state_actions = q_table.iloc[s,:]
	if(np.random.uniform()>EPSILON or ((state_actions == 0).all())):
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax()
	return action_name


def get_feedback(S,A):
	if A == 'right':
		if S == N_STATES - 4:
			S_ = 'terminal'
			R = 20
		else:
			S_ = S+1
			R = 0
	else:
		R =0
		if S == 0:
			S_ = S
		else:
			S_ = S-1
	return S_ ,R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']  
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def RL():
	q_table = Q_Table(N_STATES,ACTIONS)
	for episode in range(MAX_EPISODES):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)
		while not is_terminated:
			A = choose_action(S, q_table)
			S_, R = get_feedback(S, A)
			q_predict = q_table.loc[S, A]
			if S_ != 'terminal':
				q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
			else:
				q_target = R     # next state is terminal
				is_terminated = True    # terminate this episode

			q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
			S = S_  # move to next state


			update_env(S, episode, step_counter+1)
			step_counter += 1

	return q_table


if __name__ == "__main__":
    q_table = RL()
    print('\r\nQ-table:\n')
    print(q_table)




"""Actions = ['left','right','up','down']
state = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
a = choose_action(10,state)"""