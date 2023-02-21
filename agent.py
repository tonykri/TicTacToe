import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class AIPlayer:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.5
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.batch_size = 9
        
        self.memory_buffer= list()
        self.max_memory_buffer = 2000
        
        self.model = Sequential([
            Dense(units=35,input_dim=state_size, activation = 'relu'),
            Dense(units=20, activation = 'relu'),
            Dense(units=action_size, activation = 'softmax')
        ])
        self.model.compile(loss="categorical_crossentropy",
                      optimizer = Adam(lr=self.lr))
        
    
    
    def compute_action(self, current_state, available_actions):
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(available_actions)
        q_values = self.model.predict(current_state)[0]
        while np.argmax(q_values) not in available_actions:
            q_values[np.argmax(q_values)] = -200
        return np.argmax(q_values)

    

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
    


    def store_episode(self,current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    


    def train(self):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        
        for experience in batch_sample:
            q_current_state = self.model.predict(experience["current_state"])
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            self.model.fit(experience["current_state"], q_current_state, verbose=0)
    

